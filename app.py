import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# 1. Page Configuration
st.set_page_config(page_title="DeepRead AI", layout="wide", page_icon="📚")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("📚 DeepRead: Academic Synthesis Engine")
st.markdown("---")

# 2. API Key Setup (From Streamlit Secrets)
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Please add your GOOGLE_API_KEY to Streamlit Secrets!")
    st.stop()

# 3. Sidebar: File Upload & Auto-Glossary
with st.sidebar:
    st.header("📂 Study Materials")
    uploaded_files = st.file_uploader("Upload PDFs (Notes, Books)", type="pdf", accept_multiple_files=True)
    
    st.divider()
    st.header("🧠 Auto-Glossary")
    glossary_area = st.empty()
    glossary_area.info("Upload a PDF to generate a glossary of key terms.")

# 4. Core Logic
if uploaded_files:
    all_docs = []
    
    with st.spinner("DeepRead is indexing your documents..."):
        for file in uploaded_files:
            # Save temporary file
            with open(file.name, "wb") as f:
                f.write(file.getbuffer())
            
            loader = PyPDFLoader(file.name)
            all_docs.extend(loader.load())
            os.remove(file.name) # Clean up

        # Split text for RAG
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = text_splitter.split_documents(all_docs)

        # Initialize Gemini Embeddings & Vector Store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        
        # Initialize Gemini Model
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

    st.success(f"✅ Loaded {len(uploaded_files)} documents. You are ready for exam prep!")

    # 5. Feature: Auto-Glossary Sidebar Update
    if st.sidebar.button("Generate Glossary"):
        glossary_query = "List 5-10 technical or difficult terms from these documents and provide brief definitions for a student."
        glossary_response = llm.invoke(glossary_query + str(all_docs[:2])) # Scan first few pages
        glossary_area.markdown(glossary_response.content)

    # 6. Main Interface: Tabs
    tab1, tab2 = st.tabs(["💬 Ask your Notes", "📝 Exam Summarizer"])

    with tab1:
        query = st.text_input("Ask a question (e.g., 'Compare the two chapters on Mitochondria'):")
        if query:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm, 
                chain_type="stuff", 
                retriever=vectorstore.as_retriever(), 
                return_source_documents=True
            )
            response = qa_chain.invoke({"query": query})
            
            st.markdown("### 🤖 DeepRead Answer:")
            st.write(response["result"])
            
            # Feature: Citation Mapping
            st.markdown("### 📍 Source Citations:")
            for doc in response["source_documents"]:
                with st.expander(f"Reference: {doc.metadata['source']} (Page {doc.metadata['page']+1})"):
                    st.write(doc.page_content)

    with tab2:
        if st.button("Generate Master Study Guide"):
            with st.spinner("Synthesizing information..."):
                summary_query = "Create a structured study guide. Include: 1. Main Concepts, 2. Key Formulas/Dates, 3. Possible Exam Questions."
                # Using RAG to summarize
                summary_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
                summary_result = summary_qa.invoke({"query": summary_query})
                st.markdown(summary_result["result"])
