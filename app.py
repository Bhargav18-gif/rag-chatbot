from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

import streamlit as st

# Load PDF
loader = PyPDFLoader("mypdf.pdf")
documents = loader.load()

# Split text
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# Embeddings
embeddings = OpenAIEmbeddings()

# Store in FAISS
db = FAISS.from_documents(docs, embeddings)

# LLM
llm = ChatOpenAI()

qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# UI
st.title("📚 My Study Chatbot")

query = st.text_input("Ask your question:")

if query:
    result = qa.run(query)
    st.write(result)
