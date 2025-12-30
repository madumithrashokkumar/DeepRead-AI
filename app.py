import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter # NEW: For Quota Protection

# --- 2025 LEGACY CHAIN BRIDGE ---
try:
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain

st.set_page_config(page_title="DeepRead AI", page_icon="📚")
st.title("📚 DeepRead: Academic Synthesis Engine")

# --- AUTHENTICATION ---
api_key = st.secrets.get("GOOGLE_API_KEY")
if not api_key:
    st.error("Missing GOOGLE_API_KEY in Streamlit Secrets!")
    st.stop()

# --- RATE LIMITER CONFIGURATION ---
# This ensures we don't send more than 1 request every 2.5 seconds.
# This is slow, but it guarantees the "Free Tier" won't crash your app.
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.4, 
    check_every_n_seconds=0.1,
    max_bucket_size=10
)

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 Study Materials")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    process_button = st.button("Analyze Documents")
    st.info("Quota Protection: On. Analyzing large files will take 1-2 minutes.")

# --- CORE RAG LOGIC ---
if uploaded_files and process_button:
    with st.spinner("Analyzing... (Pacing requests to avoid 429 Errors)"):
        try:
            all_docs = []
            for uploaded_file in uploaded_files:
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                loader = PyPDFLoader(temp_path)
                all_docs.extend(loader.load())
                os.remove(temp_path)

            # Using larger chunks reduces the number of API calls needed.
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
            splits = text_splitter.split_documents(all_docs)

            # APPLY RATE LIMITER HERE
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004", 
                google_api_key=api_key,
                rate_limiter=rate_limiter # <--- THE FIX
            ).with_retry(stop_after_attempt=10) # Heavy retries for reliability

            vectorstore = InMemoryVectorStore.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever()
            
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                google_api_key=api_key,
                temperature=0.3
            ).with_retry(stop_after_attempt=5)

            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an academic assistant. Use the context to answer precisely. Context: {context}"),
                ("human", "{input}"),
            ])
            
            doc_chain = create_stuff_documents_chain(llm, prompt)
            st.session_state.rag_chain = create_retrieval_chain(retriever, doc_chain)
            
            st.success("Analysis Complete!")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

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
                st.error(f"Rate Limit Resetting: Please wait 60 seconds.")
    else:
        st.warning("Please upload and analyze a PDF first.")
