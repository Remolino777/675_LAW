import streamlit as st
from neo4j import GraphDatabase
#from src.neo4j_functions import get_related_nodes, create_neo4j_driver, get_generative_model, vector_store, llm_interaction
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Neo4jVector
import time
import markdown

st.set_page_config(  layout="wide"    
)

sistema = """
**Sistema:** Eres un asistente legal especializado en la Ley 675 de 2001 de Colombia sobre Propiedad Horizontal. Tu objetivo es proporcionar respuestas precisas y prácticas basadas en la información legal disponible.

Instrucciones para tus respuestas:

**Formato de Respuesta:**
1. Comienza con un título principal en negrita y tamaño 18px
2. El texto regular debe ser de 14px
3. Usa espaciado doble entre secciones
4. Utiliza viñetas (*) para listas
5. Resalta conceptos importantes en **negrita**

**Estructura de Respuesta:**
# [Título de la Consulta]

**Fundamento Legal:**
* Cita los artículos específicos de la Ley 675 que aplican al caso
* Explica la interpretación legal de manera clara y concisa

**Análisis del Caso:**
* Relaciona los hechos específicos con la normativa
* Proporciona una explicación detallada
* Incluye jurisprudencia relevante si está disponible

**Ejemplos Prácticos:**
* Presenta situaciones similares cuando sea posible
* Explica cómo se han resuelto casos parecidos

**Recomendaciones:**
* Ofrece sugerencias prácticas
* Menciona posibles alternativas de solución

**¿Necesitas más información?**
[Invitación cordial a realizar más consultas o solicitar aclaraciones]

Recuerda:
1. Basa tus respuestas ÚNICAMENTE en la información RAG proporcionada
2. Mantén un tono profesional pero accesible
3. Si hay ambigüedad, solicita aclaraciones
4. No hagas suposiciones fuera del marco legal
5. Indica si algún aspecto requiere consulta con un abogado especializado

¿Cómo puedo ayudarte con tu consulta sobre la Ley 675 de 2001?
"""


#Functions #######################

# Crear un driver de conexion a NEO4J.
@st.cache_resource
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

# Crea una instancia de SentenceEmbedding

@st.cache_resource
class SentenceEmbedding:
    def __init__(self, _model): #La solución es modificar la función para que Streamlit ignore el hasheo del modelo usando un guión bajo como prefijo.
        self.model = _model

    def embed_query(self, text):
        return self.model.encode(text).tolist()  # Convierte a lista

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()  # Convierte a lista
# Función para obtener nodos relacionados pregunta-[responde]->articulo
# Se nesecita para crear los embedings en el nodo . Neo4j.Vector nesecita este el model_embedding y en este caso estoy utilizando el sentence_transformer
@st.cache_resource
def vector_store(uri, username, password):
    model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
    embedding = SentenceEmbedding(_model=model)
    
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
# Función para configurar y obtener el modelo generativo
@st.cache_resource
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
# Esta funcion no nesecita ser guardada en el cache
def llm_interaction(query,vector_store, _driver, _llm):
    '''
    driver= conexion a neo4j
    vector_store=los embeddings almacenados
    query=pregunta a realizar
    llm = modelo generativo definido, en este caso sera giminie flash
    '''
    node = vector_store.similarity_search(query, k=1)
    id = node[0].metadata.get('pregunta_id')
    
    response = get_related_nodes(session=driver.session(),pregunta_id=id)
    my_prompt = f"{sistema}. Usando el siguiente contexto: {response} responde la siguiente pregunta:{query}"
    ans = llm.generate_content(my_prompt)
    
    return ans.text  

#Enviroment Variables #############################
URI = 'neo4j+s://295c5d0b.databases.neo4j.io'
NEO4J_USER = st.secrets['N_USER']
NEO4J_PASSWORD = st.secrets['N_PASSWORD']
G_KEY = st.secrets['GEMINI_API']

#Model and Database Setup###############
#Load vector store
vs = vector_store(uri=URI, username=NEO4J_USER,password=NEO4J_PASSWORD)
#Create connection to the neo4j database
driver = create_neo4j_driver(uri=URI, user=NEO4J_USER,password=NEO4J_PASSWORD)
#Load llm model ''gemini-1.5-flash-latest'' 
llm = get_generative_model(api_key=G_KEY, model_name='gemini-1.5-flash-latest', temperature=0.3, top_p=0.95, top_k=5, max_output_tokens=8192, response_mime_type="text/plain")

st.sidebar.title('Asistente legal de propiedad horizontal (Ley 675)')
st.sidebar.image('Corte_Suprema_de_Justicia_de_Colombia.svg.png')
    

#Gui interaction ######################

# Start chatbot history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(f"<div style='font-size: 16px;'>{message['content']}</div>", unsafe_allow_html=True)

# React to user input
prompt = st.chat_input('Que consulta a la ley quieres hacer?')
if prompt:
    # Mostrar mensaje del usuario
    with st.chat_message('user'):
        st.markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    
    # Generar y mostrar respuesta del asistente
    with st.chat_message('assistant'):
        message_placeholder = st.empty()
        
        # Mostrar spinner mientras se genera la respuesta
        with st.spinner('Generando respuesta...'):
            # Obtener respuesta del modelo
            response = llm_interaction(query=prompt, vector_store=vs, _driver=driver, _llm=llm)
            
            # Preparar la respuesta completa con el disclaimer
            disclaimer = "\n\n\n *Este chatbot no reemplaza el asesoramiento legal profesional ni el juicio de un abogado licenciado. La consulta con un abogado es esencial para recibir un análisis exhaustivo de cualquier situación legal.*"
            
            full_response = f"<div style='font-size: 18px;'>{response}{disclaimer}</div>"
            
            # Mostrar la respuesta completa
            message_placeholder.markdown(full_response, unsafe_allow_html=True)
            
            # Guardar en el historial (sin el disclaimer y sin el HTML)
            st.session_state.messages.append({'role': 'assistant', 'content': response})