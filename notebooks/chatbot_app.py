import streamlit as st



st.sidebar.title('Asistente legal de propiedad horizontal (Ley 675)')
st.sidebar.image('Corte_Suprema_de_Justicia_de_Colombia.svg.png')
    



# Start chatbot history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# React to user input
if prompt := st.chat_input('¿En qué te puedo ayudar?'):
    # Display user message in chat message container
    with st.chat_message('user'):
        st.markdown(prompt)
    # Append the actual user prompt, not the string 'prompt'
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    response = f'Echo: {prompt}'
    # Display assistant response in chat message container
    with st.chat_message('assistant'):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({'role': 'assistant', 'content': response})
