import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.retrieval import create_retrieval_chain

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
    st.info("Note: Using Google Free Tier. Large files may take a moment to process due to rate limits.")

# --- CORE RAG LOGIC ---
if uploaded_files and process_button:
    with st.spinner("Analyzing... (Applying Rate-Limit Protection)"):
        try:
            all_docs = []
            for uploaded_file in uploaded_files:
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                loader = PyPDFLoader(temp_path)
                all_docs.extend(loader.load())
                os.remove(temp_path)

            # 1. Increased chunk size to reduce total API calls (Stops 429 Errors)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
            splits = text_splitter.split_documents(all_docs)

            # 2. Added .with_retry() to automatically wait if Google says "Too Fast"
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004", 
                google_api_key=api_key
            ).with_retry(
                stop_after_attempt=6,
                wait_exponential_jitter=True
            )

            vectorstore = InMemoryVectorStore.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever()
            
            # 3. Use standard Flash for better free-tier stability
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
            
            st.success("Analysis Complete! You can now ask questions.")
            
        except Exception as e:
            if "429" in str(e):
                st.error("Google's Rate Limit hit. Please wait 60 seconds and try again. The app is designed to auto-retry, but the document may be too large for a single minute.")
            else:
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
                st.error("Quota exceeded for this minute. Please wait 60 seconds.")
    else:
        st.warning("Please upload and analyze a PDF first.")
