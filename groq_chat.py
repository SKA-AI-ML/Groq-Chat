from groq import Groq
import streamlit as st

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq


def buildUI():
    st.session_state.model = st.sidebar.selectbox('Choose a model', ['llama3-8b-8192', 'llama3-70b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it'])

    st.session_state.conversation_memory_length = st.sidebar.slider('Conversational Memory Length:', 1, 30, value=10)

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role":"assistant", "content":"How can i help you?"}]
    
    for message in st.session_state.messages:
        st.chat_message(message["role"]).write(message["content"])
    
    


def buildAI():
    memory = ConversationBufferWindowMemory(k=st.session_state.conversation_memory_length)
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    for message in st.session_state.chat_history:
        memory.save_context({'input':message['human']}, {'output':message['AI']})

    groq_chat = ChatGroq(groq_api_key=st.session_state.groq_api_key, model_name=st.session_state.model)

    conversation = ConversationChain(llm=groq_chat, memory=memory)

    if prompt := st.chat_input():
        st.session_state.messages.append({"role":"user", "content":prompt})
        st.chat_message("user").write(prompt)

        response = conversation(prompt)
        message = {'human':prompt, 'AI':response['response']}
        st.session_state.chat_history.append(message)

        st.session_state.messages.append({"role":"assistant", "content":response['response']})
        st.chat_message("assistant").write(response['response'])

        
def main():

    st.title("💬 Groq Chat")
    st.sidebar.title("Settings")
    token = st.sidebar.text_input("Enter Groq token here...")

    if token:
        if token != "":
            st.session_state.groq_api_key = token
            buildUI()
            buildAI()


if __name__ == "__main__":
    main()
