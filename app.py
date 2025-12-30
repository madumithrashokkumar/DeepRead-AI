import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import InMemoryVectorStore  # Changed path
from langchain.chains import RetrievalQA

# --- APP CONFIGURATION ---
st.set_page_config(page_title="DeepRead AI", page_icon="📚")
st.title("📚 DeepRead: Academic Synthesis Engine")

# --- AUTHENTICATION ---
api_key = st.secrets.get("GOOGLE_API_KEY")
if not api_key:
    st.error("Missing GOOGLE_API_KEY in Secrets!")
    st.stop()

# --- SIDEBAR: PDF UPLOAD ---
with st.sidebar:
    st.header("📂 Study Materials")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    process_button = st.button("Analyze Documents")

# --- CORE RAG LOGIC ---
if uploaded_files and process_button:
    with st.spinner("Analyzing..."):
        try:
            all_docs = []
            for uploaded_file in uploaded_files:
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                loader = PyPDFLoader(temp_path)
                all_docs.extend(loader.load())
                os.remove(temp_path)

            # Split text
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(all_docs)

            # Create Embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004", 
                google_api_key=api_key
            )

            # CREATE VECTOR STORE (Simplified)
            vectorstore = InMemoryVectorStore.from_documents(
                documents=splits, 
                embedding=embeddings
            )
            
            # Initialize LLM & Chain
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b", google_api_key=api_key)
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
            )
            st.success("Ready! Ask your questions below.")
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

# --- CHAT INTERFACE ---
st.divider()
query = st.text_input("Ask a question about your documents:")
if query:
    if "qa_chain" in st.session_state:
        with st.spinner("Thinking..."):
            response = st.session_state.qa_chain.invoke(query)
            st.markdown(f"### 🤖 Answer:\n{response['result']}")
    else:
        st.warning("Please analyze a PDF first.")
