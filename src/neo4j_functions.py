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
    
def llm_interaction(query: str, vector_store, driver, llm) -> str:
    '''
    Función que procesa una consulta legal utilizando Neo4j y un modelo de lenguaje

    Args:
        query (str): Pregunta del usuario
        vector_store: Vector store con los embeddings almacenados
        driver: Conexión a Neo4j
        llm: Modelo generativo (Gemini)

    Returns:
        str: Respuesta generada por el modelo
    
    Raises:
        ValueError: Si no se encuentran resultados similares
        Exception: Para otros errores durante el proceso
    '''
    try:
        # Realizar búsqueda de similaridad
        results = vector_store.similarity_search_with_score(query, k=1)
        
        if not results:
            raise ValueError("No se encontraron resultados similares")
        
        # Desempaquetar el primer resultado
        doc, similarity_score = results[0]
        
        # Verificar si el score está por debajo de un umbral mínimo
        MIN_SIMILARITY_THRESHOLD = 0.8
        if similarity_score < MIN_SIMILARITY_THRESHOLD:
            print(f"Advertencia: Baja similaridad ({similarity_score})")
            response ='''
            Recuerda que tu pregunta tiene que estar relacionada con la ley 675 de Colombia. 
            Ademas procura hacer una pregunta a la vez para darte una respuesta mas clara
            '''
            return response
        
        # Obtener el ID de la pregunta
        pregunta_id = doc.metadata.get('pregunta_id')
        if not pregunta_id:
            raise ValueError("No se encontró el ID de la pregunta en los metadatos")
            
        print(f"ID Pregunta: {pregunta_id}, Score: {similarity_score}")
        
        # Obtener nodos relacionados de Neo4j
        with driver.session() as session:
            contexto = get_related_nodes(session=session, pregunta_id=pregunta_id)
            
            if not contexto:
                raise ValueError(f"No se encontró contexto para la pregunta ID {pregunta_id}")
        
        # Construir el prompt
        prompt = f"""Sos un asistente legal especializado en la ley 675 de Colombia.
        
Contexto relevante:
{contexto}

Pregunta del usuario:
{query}

Por favor, proporciona una respuesta detallada. Si es aplicable, incluye ejemplos prácticos.
Al finalizar, invita al usuario a realizar más preguntas sobre la ley 675."""

        # Generar respuesta
        response = llm.generate_content(prompt)
        
        return response.text
        
    except ValueError as ve:
        return f"Lo siento, no pude procesar tu pregunta: {str(ve)}"
    except Exception as e:
        print(f"Error en llm_interaction: {str(e)}")
        return "Lo siento, ocurrió un error al procesar tu pregunta. Por favor, intenta reformularla."
   

