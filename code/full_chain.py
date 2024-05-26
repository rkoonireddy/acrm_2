import os
from dotenv import load_dotenv
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceHub
from langchain_community.chat_models.huggingface import ChatHuggingFace

from dotenv import load_dotenv

from basic_chain import get_model
from memory import create_memory_chain
from rag_chain import make_rag_chain

MISTRAL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
ZEPHYR_ID = "HuggingFaceH4/zephyr-7b-beta"
# LLAMA_ID = "meta-llama/Meta-Llama-3-8B"
# ZEPHYR_ID_2 = "HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1"

def create_full_chain(retriever, system_prompt=None, openai_api_key=None, chat_memory=ChatMessageHistory()):
    if not system_prompt:
        system_prompt = """You help credit risk officers to evaluate a loan application. Based on the details given to you about a person or a loan application, you suggest giving them loan or not and why you arrived at that conclusion.
        Use the following context and the users' chat history to help the user:
        If you don't know the answer, just say that you don't know. 
        
        Context: {context}
        
        Question: """

    model = get_model("ChatGPT", openai_api_key=openai_api_key)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    rag_chain = make_rag_chain(model, retriever, rag_prompt=prompt)
    chain = create_memory_chain(model, rag_chain, chat_memory)
    return chain


def ask_question(chain, query):
    response = chain.invoke(
        {"question": query},
        config={"configurable": {"session_id": "foo"}}
    )
    return response
