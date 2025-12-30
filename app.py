import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA  # Using the classic chain for stability

# --- APP CONFIGURATION ---
st.set_page_config(page_title="DeepRead AI", page_icon="📚", layout="wide")
st.title("📚 DeepRead: Academic Synthesis Engine")

# --- AUTHENTICATION ---
# This pulls your new AI Studio key from Streamlit Secrets
api_key = st.secrets.get("GOOGLE_API_KEY")

if not api_key:
    st.error("Please add your GOOGLE_API_KEY to Streamlit Secrets!")
    st.stop()

# --- SIDEBAR: PDF UPLOAD ---
with st.sidebar:
    st.header("📂 Study Materials")
    uploaded_files = st.file_uploader("Upload PDFs (Notes, Books)", type="pdf", accept_multiple_files=True)
    process_button = st.button("Analyze Documents")

# --- CORE RAG LOGIC ---
if uploaded_files and process_button:
    with st.spinner("Processing documents into knowledge..."):
        all_docs = []
        
        # 1. Load and read PDFs
        for uploaded_file in uploaded_files:
            # Save temporary file to read it
            temp_file = f"./temp_{uploaded_file.name}"
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            loader = PyPDFLoader(temp_file)
            all_docs.extend(loader.load())
            os.remove(temp_file) # Clean up

        # 2. Split text into chunks (Optimized size for free tier)
        text_splitter = RecursiveCharacterCharacterSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(all_docs)

        # 3. Create Embeddings (Updated model for 2025)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", 
            google_api_key=api_key
        )

        # 4. Create Vector Store (In-memory for Streamlit Cloud)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        
        # 5. Initialize LLM (Using the lighter 8b model to avoid quota errors)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-8b", 
            google_api_key=api_key,
            temperature=0.3
        )

        # 6. Setup the Retrieval Chain
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )
        st.success("Analysis Complete! You can now ask questions.")

# --- CHAT INTERFACE ---
st.divider()
query = st.text_input("Ask a question about your study materials:")

if query:
    if "qa_chain" in st.session_state:
        with st.spinner("Finding answer..."):
            response = st.session_state.qa_chain.invoke(query)
            st.markdown("### 🤖 DeepRead Analysis:")
            st.write(response["result"])
    else:
        st.warning("Please upload and process a PDF first!")

# --- FOOTER ---
st.markdown("---")
st.caption("Powered by Google Gemini 1.5 Flash-8B & LangChain")
