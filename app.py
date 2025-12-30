import streamlit as st
import os
import time
import random
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate

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

# --- RETRY DECORATOR ---
# This automatically waits and retries if Google says "Resource Exhausted"
@retry(
    retry=retry_if_exception_type(Exception), 
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    reraise=True
)
def add_to_vectorstore_with_retry(vectorstore, batch):
    return vectorstore.add_documents(batch)

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 Study Materials")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    process_button = st.button("Analyze Documents")
    st.info("💡 **Tip:** Smaller batches are safer for Free Tier. Processing will pause if API limits are hit.")

# --- CORE RAG LOGIC ---
if uploaded_files and process_button:
    with st.spinner("Analyzing and Pacing..."):
        try:
            # 1. Load and Split
            all_docs = []
            for uploaded_file in uploaded_files:
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                loader = PyPDFLoader(temp_path)
                all_docs.extend(loader.load())
                os.remove(temp_path)

            # Larger chunks = fewer total API requests (The most effective 429 fix)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)
            splits = text_splitter.split_documents(all_docs)

            # 2. Setup Embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004", 
                google_api_key=api_key,
                task_type="retrieval_document"
            )

            # 3. Dynamic Batching with Backoff
            vectorstore = InMemoryVectorStore(embedding=embeddings)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Small batch size (3) prevents hitting the "Tokens Per Minute" limit
            batch_size = 3 
            for i in range(0, len(splits), batch_size):
                batch = splits[i : i + batch_size]
                
                try:
                    add_to_vectorstore_with_retry(vectorstore, batch)
                except Exception as e:
                    if "429" in str(e):
                        status_text.warning("API Quota Hit. Cool-down active... (Waiting 60s)")
                        time.sleep(60) # Hard reset wait for Free Tier
                        add_to_vectorstore_with_retry(vectorstore, batch)
                
                # Update UI
                percent = min((i + batch_size) / len(splits), 1.0)
                progress_bar.progress(percent)
                status_text.text(f"Processed {min(i+batch_size, len(splits))}/{len(splits)} chunks...")
                
                # "Jittered" sleep: Random wait between 3-6 seconds makes requests less "robotic"
                time.sleep(random.uniform(3, 6))

            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            
            # 4. LLM Setup (Flash is faster and has higher Free limits than Pro)
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                google_api_key=api_key,
                temperature=0.2
            ).with_retry() # LangChain's native retry for chat

            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an academic assistant. Context: {context}"),
                ("human", "{input}"),
            ])
            
            doc_chain = create_stuff_documents_chain(llm, prompt)
            st.session_state.rag_chain = create_retrieval_chain(retriever, doc_chain)
            
            status_text.empty()
            progress_bar.empty()
            st.success("Analysis Complete!")
            
        except Exception as e:
            st.error(f"Critical Error: {str(e)}")

# --- CHAT INTERFACE ---
st.divider()
query = st.text_input("Ask a question about your documents:")
if query:
    if "rag_chain" in st.session_state:
        with st.spinner("Searching and Reasoning..."):
            try:
                response = st.session_state.rag_chain.invoke({"input": query})
                st.markdown(f"### 🤖 Answer:\n{response['answer']}")
            except Exception as e:
                st.error("The API is busy. Please wait 15-30 seconds and try again.")
    else:
        st.warning("Please upload and analyze a PDF first.")
