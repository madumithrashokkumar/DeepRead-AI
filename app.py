import streamlit as st
import os

# --- 2025 IMPORT FIX ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate

# Import chains from the classic/legacy bridge
try:
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    # Fallback for environments where classic is integrated
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain

# --- APP CONFIGURATION ---
st.set_page_config(page_title="DeepRead AI", page_icon="📚")
st.title("📚 DeepRead: Academic Synthesis Engine")

# --- AUTHENTICATION ---
api_key = st.secrets.get("GOOGLE_API_KEY")
if not api_key:
    st.error("Missing GOOGLE_API_KEY in Streamlit Secrets!")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 Study Materials")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    process_button = st.button("Analyze Documents")
    st.info("Using Gemini 1.5 Flash (Free Tier)")

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

            # Increased chunk size slightly to reduce API calls
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=250)
            splits = text_splitter.split_documents(all_docs)

            # FIXED: Removed .with_retry() and added native request_options for 2025 compatibility
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004", 
                google_api_key=api_key,
                task_type="retrieval_document"
            )

            vectorstore = InMemoryVectorStore.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever()
            
            # LLM supports .with_retry() natively in 2025
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                google_api_key=api_key,
                temperature=0.3
            ).with_retry(stop_after_attempt=3)

            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an academic assistant. Use the context to answer precisely. Context: {context}"),
                ("human", "{input}"),
            ])
            
            doc_chain = create_stuff_documents_chain(llm, prompt)
            st.session_state.rag_chain = create_retrieval_chain(retriever, doc_chain)
            
            st.success("Analysis Complete!")
            
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")

# --- CHAT INTERFACE ---
st.divider()
query = st.text_input("Ask a question about your documents:")
if query:
    if "rag_chain" in st.session_state:
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.rag_chain.invoke({"input": query})
                st.markdown(f"### 🤖 Answer:\n{response['answer']}")
            except Exception as e:
                st.error(f"Quota error: {str(e)}. Please wait 60 seconds for the API limit to reset.")
    else:
        st.warning("Please upload and analyze a PDF first.")
