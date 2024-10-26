from neo4j import GraphDatabase

import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession


#Model import
import google.generativeai as genai 
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Neo4jVector




# Crear un driver de conexion a NEO4J.
def create_neo4j_driver(uri, user, password):
    """
    Crea un driver de conexión a Neo4j.

    :param uri: URI de la base de datos Neo4j.
    :param user: Usuario para la autenticación.
    :param password: Contraseña para la autenticación.
    :return: Un driver de Neo4j listo para ser usado.
    """
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        return driver
    except Exception as e:
        print(f"Error al crear el driver de Neo4j: {e}")
        return None
#_______________
# Función para obtener nodos relacionados pregunta-[responde]->articulo
def get_related_nodes(session, pregunta_id):
    if not pregunta_id:
        raise ValueError("Pregunta_id no puede ser None o vacío.")    
    
    # Buscar en todos los nodos preguntas
    query = """
    MATCH (start:preguntas)   
    WHERE start.pregunta_id = $pregunta_id
    CALL apoc.path.expand(
        start,              // nodo inicial
        'responde>',        // tipo de relación
        '',                 // filtro de etiquetas (vacío para aceptar todas)
        1,                  // profundidad mínima
        1                   // profundidad máxima (antes era maxLevel)
    )
    YIELD path
    RETURN last(nodes(path)).articulo_name AS articulo_name,
           last(nodes(path)).contenido AS contenido_final
    """
    
    try:
        result = session.run(query, pregunta_id=pregunta_id)
        record = next(result) if result else None
        if record:
            text = f'articulo_numero: {record["articulo_name"]}\n{record["contenido_final"]}'
            return text  # 
        return ""
    except Exception as e:
        print(f"Error ejecutando la consulta: {e}")
        return ""


#__________________
# Función para configurar y obtener el modelo generativo
def get_generative_model(api_key, model_name='gemini-1.5-flash-latest', temperature=0.3, top_p=0.95, top_k=5, max_output_tokens=8192, response_mime_type="text/plain"):
    """
    Configura y devuelve un modelo generativo con la API de Gemini.

    :param api_key: La clave de API para acceder al servicio.
    :param model_name: Nombre del modelo generativo (por defecto 'gemini-1.5-flash-latest').
    :param temperature: Controla la aleatoriedad en la generación (por defecto 0.3).
    :param top_p: El porcentaje de probabilidad para la selección de tokens (por defecto 0.95).
    :param top_k: El número de mejores tokens a considerar (por defecto 5).
    :param max_output_tokens: Número máximo de tokens que puede generar el modelo (por defecto 8192).
    :param response_mime_type: El tipo de respuesta del modelo (por defecto 'text/plain').
    :return: El modelo generativo configurado.
    """
    try:
        # Configuración de la API
        genai.configure(api_key=api_key)
        
        # Configuración para la generación
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_output_tokens": max_output_tokens,
            "response_mime_type": response_mime_type,
        }
        
        # Crear el modelo generativo
        llm = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
        
        return llm
    except Exception as e:
        print(f"Error al configurar el modelo generativo: {e}")
        return None


# Modelo optimizado para trabajar en espanol y para hacer preguntas.

# Se nesecita para crear los embedings en el nodo . Neo4j.Vector nesecita este el model_embedding y en este caso estoy utilizando el sentence_transformer
class SentenceEmbedding:
    def __init__(self, model):
        self.model = model

    def embed_query(self, text):
        return self.model.encode(text).tolist()  # Convierte a lista

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()  # Convierte a lista
    
# Crea una instancia de SentenceEmbedding

def vector_store(uri, username, password):
    model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
    embedding = SentenceEmbedding(model)
    
    vector_store=Neo4jVector.from_existing_graph(
    embedding=embedding,
    url=uri,
    username=username,
    password=password,
    index_name="p_vector_index",
    node_label="preguntas",
    text_node_properties=["pregunta"],
    embedding_node_property="pregunta_embedding",
    )
    return vector_store
    
def llm_interaction(query, vector_store, driver, llm):
    '''
    driver= conexion a neo4j
    vector_store=los embeddings almacenados
    query=pregunta a realizar
    llm = modelo generativo definido, en este caso sera giminie flash
    '''
    node = vector_store.similarity_search(query, k=1)
    id = node[0].metadata.get('pregunta_id')
    print(id)
    response = get_related_nodes(session=driver.session(),pregunta_id=id)
    my_prompt = f"Sos un asistente legal especializado en la ley 675 de Colombia y usando el siguiente contexto: {response} responde la siguiente pregunta:{query} de la manera mas detallada posible y si aplica utiliza algun ejemplo y al finalizar invita al usuario a volver a preguntar"
    ans = llm.generate_content(my_prompt)
    return ans.text    

