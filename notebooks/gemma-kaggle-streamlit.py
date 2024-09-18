import streamlit as st
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configurar la API de Kaggle
os.environ['KAGGLE_USERNAME'] = "tu_usuario_de_kaggle"
os.environ['KAGGLE_KEY'] = "tu_clave_de_api_de_kaggle"

# Inicializar la API de Kaggle
api = KaggleApi()
api.authenticate()

# Descargar el modelo de Kaggle si no existe localmente
model_path = "./gemma-2b-it"
if not os.path.exists(model_path):
    api.dataset_download_files('google/gemma-2b-it', path='.', unzip=True)

# Cargar el modelo y el tokenizador
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)

# Función para generar respuestas
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Interfaz de Streamlit
st.title("Chatbot con Gemma 2B")

# Inicializar el historial de chat si no existe
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes del chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input del usuario
if prompt := st.chat_input("Escribe tu mensaje aquí"):
    # Agregar mensaje del usuario al historial
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar respuesta
    response = generate_response(prompt)

    # Agregar respuesta al historial y mostrarla
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
