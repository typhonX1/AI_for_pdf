import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

GOOGLE_API_KEY = "AIzaSyDEbMTwl6pvaODmLCIuaszVZJe3J_R3lBA"  
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

st.set_page_config(page_title="📄 AI PDF Q&A", layout="wide")
st.title("📄 AI PDF Question Answering System")
st.markdown("Upload a PDF, process it, and ask questions!")

uploaded_file = st.file_uploader("📂 Upload your PDF", type="pdf")

if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)
    pdf_text = "".join([page.extract_text() or "" for page in pdf_reader.pages])

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(pdf_text)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    vectorstore = FAISS.from_texts(texts, embeddings)

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.2)

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(), chain_type="stuff")

    user_question = st.text_input("🔎 Ask a question from the PDF:")

    if user_question:
        answer = qa_chain.run(user_question)
        st.write("**Answer:**", answer)
