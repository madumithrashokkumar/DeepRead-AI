import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
# --- FIX IS HERE ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
# -------------------
from langchain.chains import RetrievalQA

# Rest of the code remains the same...

# 1. UI Setup
st.set_page_config(page_title="DeepRead AI", layout="wide")
st.title("📚 DeepRead: Your Academic Research Copilot")

# Sidebar for Uploads and Auto-Glossary
with st.sidebar:
    st.header("Upload Study Materials")
    uploaded_files = st.file_uploader("Upload PDFs (Notes, Textbooks)", accept_multiple_files=True)
    st.divider()
    st.header("📝 Auto-Glossary")
    glossary_placeholder = st.empty()

# 2. Processing Logic
if uploaded_files:
    all_docs = []
    for file in uploaded_files:
        # Save temp file to load it
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())
        loader = PyPDFLoader(file.name)
        all_docs.extend(loader.load())

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(all_docs)

    # Create Vector Store
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    # Define the AI Model
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # 3. Main Chat Interface
    query = st.text_input("Ask a question across all documents:")
    
    if query:
        # Retrieval-Augmented Generation (RAG)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
        result = qa_chain.invoke({"query": query})
        
        st.subheader("DeepRead Answer:")
        st.write(result["result"])

        # Feature: Citation Mapping
        st.subheader("📍 Sources & Citations:")
        for doc in result["source_documents"]:
            st.info(f"Source: {doc.metadata['source']} | Page: {doc.metadata['page'] + 1}")

    # Feature: Summarizer Mode (Button)
    if st.button("Generate Exam Summary"):
        summary_prompt = "Summarize the key concepts, formulas, and definitions from these notes for an exam."
        summary_result = qa_chain.invoke({"query": summary_prompt})
        st.success(summary_result["result"])
