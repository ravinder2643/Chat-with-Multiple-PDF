import streamlit as st # type: ignore
from PyPDF2 import PdfReader # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings # type: ignore
import google.generativeai as genai # type: ignore
from langchain.vectorstores import FAISS # type: ignore
from langchain_google_genai import ChatGoogleGenerativeAI # type: ignore
from langchain.chains.question_answering import load_qa_chain # type: ignore
from langchain.prompts import PromptTemplate # type: ignore
from dotenv import load_dotenv # type: ignore

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))






def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


