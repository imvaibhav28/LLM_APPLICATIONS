# import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.llms import Ollama

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader

# from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from urllib.parse import urlparse, urlunparse
from langchain.schema.output_parser import StrOutputParser

loader = WebBaseLoader("https://fundingservice.org.uk")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter()
document_chunks = text_splitter.split_documents(docs)


from langchain_community.embeddings import GPT4AllEmbeddings

gpt4all_embd = GPT4AllEmbeddings()

vector_store = Chroma.from_documents(document_chunks, gpt4all_embd)

llm = Ollama(model="mistral")

retriever = vector_store.as_retriever()

prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context: \n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

conversation_rag_chain = create_retrieval_chain(retriever_chain, stuff_documents_chain)

from langchain.chains import ConversationalRetrievalChain

chain = ConversationalRetrievalChain.from_llm(llm, retriever,output_key='answer')

hist=["This is chat history"]

from typing import List
from fastapi import FastAPI
from langchain.llms import Ollama
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langserve import add_routes
import uvicorn
from langchain_core.output_parsers import JsonOutputParser

app = FastAPI(title="LangChain", version="1.0", description="Chatbot")

add_routes(app, chain)
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
