import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os 
from dotenv import load_dotenv
load_dotenv()
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
import torch
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.embeddings import HuggingFaceEmbeddings
## Load the Groq API key

#os.environ['groq_api'] = os.getenv('groq_api')

#groq_api_key = os.getenv("groq_api")
llm = ChatGroq(
    groq_api_key="Change with your Groq API Key",
    model_name="Llama3-8b-8192"
)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the Questions based on the provided context only.
    Please Provide the most accurate respones based on the questions
    <context>
    {context}
    <context>
    Question:{input}
    """
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        try:
            st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            directory_path = r"A:/AI_PROJECT/M.tech 2nd Semester"
            st.session_state.loader = PyPDFDirectoryLoader(directory_path)  # Data ingestion step
            st.session_state.docs = st.session_state.loader.load()  # Document loading

            if not st.session_state.docs:
                st.error("No documents found in the specified directory.")
                return

            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])

            if not st.session_state.final_documents:
                st.error("No documents available after splitting.")
                return

            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
            st.success("Vector database created successfully.")
        except Exception as e:
            st.error(f"Failed to initialize vector database: {e}")

        #st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

user_prompt = st.text_input("Enter your query from the Material")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")

if user_prompt:
    document_chain = create_stuff_documents_chain(llm,prompt)
    restriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(restriever,document_chain)
    response = retriever_chain.invoke({'input':user_prompt})
    st.write(response['answer'])

    ## With a streamlit expander
    with st.expander("Document similarity search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('-------------------')











