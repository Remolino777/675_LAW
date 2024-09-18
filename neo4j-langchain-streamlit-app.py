import streamlit as st
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Neo4jVector
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Configuración de Neo4j
NEO4J_URI = "neo4j+s://295c5d0b.databases.neo4j.io"
NEO4J_USER = "gatoyote"
NEO4J_PASSWORD = "Anarquia_1501"

# Configuración del modelo
MODEL_NAME = "google/gemma-7b"

# Inicialización del modelo y tokenizador
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=100)
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

# Inicialización de embeddings
@st.cache_resource
def load_embeddings():
    embeddings = HuggingFaceEmbeddings()
    return embeddings

# Configuración de Neo4jVector
@st.cache_resource
def setup_neo4j_vector():
    embeddings = load_embeddings()
    neo4j_vector = Neo4jVector.from_existing_index(
        embeddings,
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        index_name="your_index_name",  # Reemplaza con el nombre de tu índice
        node_label="YourNodeLabel",    # Reemplaza con la etiqueta de tus nodos
        text_node_property="description",  # Reemplaza con la propiedad que contiene el texto
        embedding_node_property="embedding",  # Reemplaza con la propiedad que almacena los embeddings
    )
    return neo4j_vector

# Configuración de la cadena de conversación
def setup_conversation_chain():
    llm = load_model()
    neo4j_vector = setup_neo4j_vector()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    template = """Usa la siguiente información para responder la pregunta del humano:
    {context}
    Pregunta del humano: {question}
    Respuesta útil:"""

    PROMPT = PromptTemplate(
        template=template, input_variables=["context", "question"]
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=neo4j_vector.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
    
    return chain

# Interfaz de Streamlit
def main():
    st.title("Chatbot Neo4j-Langchain")

    # Inicializar la cadena de conversación
    if 'conversation_chain' not in st.session_state:
        st.session_state.conversation_chain = setup_conversation_chain()

    # Mostrar mensajes del chat
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Campo de entrada para el usuario
    if prompt := st.chat_input("Escribe tu mensaje aquí"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Generar respuesta
            response = st.session_state.conversation_chain({"question": prompt})
            full_response = response['answer']
            
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
