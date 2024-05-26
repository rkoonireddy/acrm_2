import os
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.embeddings import OpenAIEmbeddings
from ensemble import create_ensemble_retriever
from full_chain import create_full_chain, ask_question
from data_loader import *
from streamlit_option_menu import option_menu

st.set_page_config(page_title="All knowing Credit Risk Manager")
st.title("All knowing Credit Risk Manager")

DATA_DIR = "./data"

def show_ui(selected_option, qa, prompt_to_user="How may I help you?"):
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": prompt_to_user}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ask_question(qa, prompt)
                st.markdown(response.content)
        message = {"role": "assistant", "content": response.content}
        st.session_state.messages.append(message)

def get_secret_or_input(secret_key, secret_name, info_link=None):
    if secret_key in st.secrets:
        secret_value = st.secrets[secret_key]
    else:
        st.write(f"Please provide your {secret_name}")
        secret_value = st.text_input(secret_name, key=f"input_{secret_key}", type="password")
        if secret_value:
            st.session_state[secret_key] = secret_value
        if info_link:
            st.markdown(f"[Get an {secret_name}]({info_link})")
    return secret_value

def save_uploaded_file(uploaded_file):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    file_path = os.path.join(DATA_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

@st.cache_resource
def get_retriever(openai_api_key=None):
    docs = load_files("./data")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small")
    return create_ensemble_retriever(docs, embeddings=embeddings)


def get_chain(openai_api_key=None, huggingfacehub_api_token=None):
    ensemble_retriever = get_retriever(openai_api_key=openai_api_key)
    chain = create_full_chain(ensemble_retriever,
                              openai_api_key=openai_api_key,
                              chat_memory=StreamlitChatMessageHistory(key="langchain_messages"))
    return chain

def get_system_prompt(selected_option):
    default_prompt = """You help credit risk officers to evaluate a loan application. Based on the details given to you about a person or a loan application, you suggest giving them loan or not and why you arrived at that conclusion.
    Use the following context and the users' chat history to help the user:
    If you don't know the answer, just say that you don't know. 
    
    Context: {context}
    
    Question: """
    if selected_option == "Credit Card Approval":
        system_prompt = """You help credit card issuers to evaluate a credit card application. Based on the details given to you about a person or an application, you suggest approving the credit card or not and why you arrived at that conclusion.
        Use the following context and the user's chat history to help the user:
        If you don't know the answer, just say that you don't know. 
        
        Context: {context}
        
        Question: """
    elif selected_option == "Loan Approval Prediction":
        system_prompt = """You help loan officers to evaluate a loan application. Based on the details given to you about a person or a loan application, you suggest giving them a loan or not and why you arrived at that conclusion.
        Use the following context and the user's chat history to help the user:
        If you don't know the answer, just say that you don't know. 
        
        Context: {context}
        
        Question: """
    # Define prompts for other options similarly
    
    return system_prompt or default_prompt

def run():
    ready = True

    openai_api_key = st.session_state.get("OPENAI_API_KEY")
    huggingfacehub_api_token = st.session_state.get("HUGGINGFACEHUB_API_TOKEN")

    with st.sidebar:
        if not openai_api_key:
            openai_api_key = get_secret_or_input('OPENAI_API_KEY', "OpenAI API key",
                                                 info_link="https://platform.openai.com/account/api-keys")
        if not huggingfacehub_api_token:
            huggingfacehub_api_token = get_secret_or_input('HUGGINGFACEHUB_API_TOKEN', "HuggingFace Hub API Token",
                                                           info_link="https://huggingface.co/docs/huggingface_hub/main/en/quick-start#authentication")

    if not openai_api_key:
        st.warning("Missing OPENAI_API_KEY")
        ready = False
    if not huggingfacehub_api_token:
        st.warning("Missing HUGGINGFACEHUB_API_TOKEN")
        ready = False

    if ready:
        chain = get_chain(openai_api_key=openai_api_key, huggingfacehub_api_token=huggingfacehub_api_token)
        st.subheader("I can predict about credit risk")
        show_ui(chain, "What would you like to know?")
    else:
        st.stop()
run()