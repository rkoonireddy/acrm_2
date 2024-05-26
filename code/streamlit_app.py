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

def get_chain(selected_option, openai_api_key=None, huggingfacehub_api_token=None, ensemble_retriever=None):
    system_prompt = get_system_prompt(selected_option)
    chain = create_full_chain(ensemble_retriever,
                              system_prompt=system_prompt,
                              openai_api_key=openai_api_key,
                              chat_memory=StreamlitChatMessageHistory(key="langchain_messages"))
    return chain

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
def get_retriever(_docs, openai_api_key=None):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-ada-002")
    return create_ensemble_retriever(_docs, embeddings=embeddings)

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
    openai_api_key = st.session_state.get("OPENAI_API_KEY")
    huggingfacehub_api_token = st.session_state.get("HUGGINGFACEHUB_API_TOKEN")
    
    with st.sidebar:
        if not openai_api_key:
            openai_api_key = get_secret_or_input('OPENAI_API_KEY', "OpenAI API key",
                                                 info_link="https://platform.openai.com/account/api-keys")
        if not huggingfacehub_api_token:
            huggingfacehub_api_token = get_secret_or_input('HUGGINGFACEHUB_API_TOKEN', "HuggingFace Hub API Token",
                                                           info_link="https://huggingface.co/docs/huggingface_hub/main/en/quick-start#authentication")
    
    with st.sidebar:
        selected_option = option_menu("Select", ["Credit Card Approval", "Loan Approval Prediction", "Risk Analysis Dashboard", "Fraud Detection", "Credit Policy Optimization"], 
            icons=['credit-card', 'cash', 'bar-chart', 'bricks', 'gear'], menu_icon="menu",default_index=0)
    
    with st.sidebar:
        selected_mode = st.sidebar.radio("Mode", ["Offline", "Online"])

        if selected_mode == "Offline":
            st.sidebar.subheader("Offline Mode")
            data_dir = DATA_DIR
            files = os.listdir(data_dir) if os.path.exists(data_dir) else []
            uploaded_file = st.sidebar.file_uploader("Upload a file")

            if not files and not uploaded_file:
                st.error(f"No files found in {data_dir}")
                st.stop()

            if uploaded_file is not None:
                if st.sidebar.button("Upload and Run"):
                    file_path = save_uploaded_file(uploaded_file)
                    st.session_state['docs'] = load_files(data_dir)
                    st.sidebar.success(f"Uploaded and processed file: {uploaded_file.name}")
                    st.rerun()

            selected_file = st.sidebar.selectbox("Select a file to view or remove", files)

            if selected_file:
                if st.sidebar.button("Remove file"):
                    os.remove(os.path.join(data_dir, selected_file))
                    st.sidebar.success(f"Removed file: {selected_file}")
                    files = os.listdir(data_dir) if os.path.exists(data_dir) else []
                    st.session_state['docs'] = load_files(data_dir)
                    st.rerun()

        elif selected_mode == "Online":
            st.sidebar.subheader("Online Mode")
            url = st.sidebar.text_input("Enter URL of the file")
            file_type = st.sidebar.selectbox("Select file type", ['PDF', 'Web Page'])
            if st.sidebar.button("Load"):
                if url:
                    try:
                        download_path = download_file(url, file_type)
                        st.session_state['docs'] = load_files(DATA_DIR)
                        st.sidebar.success(f"Loaded and processed file from URL: {url}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading file from URL: {e}")

    if not openai_api_key:
        st.warning("Missing OPENAI_API_KEY")
        st.stop()
    if not huggingfacehub_api_token:
        st.warning("Missing HUGGINGFACEHUB_API_TOKEN")
        st.stop()
    if "selected_option" not in st.session_state:
        st.session_state["selected_option"] = None
        st.session_state['docs'] = load_files(DATA_DIR)

    selected_option = selected_option.lower().capitalize() if selected_option else None
    prompt = f"I want to do a {selected_option} evaluation." if selected_option else "Please select an option."
    docs = st.session_state.get('docs', [])

    if selected_option:
        if docs:
            retriever = get_retriever(docs, openai_api_key=openai_api_key)
            chain = get_chain(selected_option, openai_api_key=openai_api_key, huggingfacehub_api_token=huggingfacehub_api_token, ensemble_retriever=retriever)
            st.subheader("I can predict about credit risk")
            show_ui(selected_option, chain, prompt)
        else:
            st.warning("No documents available for retrieval.")
    else:
        st.warning("No option selected.")


run()