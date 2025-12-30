import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
# THE FIX: Direct imports from sub-packages
from langchain.chains.combine_documents import create_stuff_documents_chain
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

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(all_docs)

            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004", 
                google_api_key=api_key
            )

            vectorstore = InMemoryVectorStore.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever()
            
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b", google_api_key=api_key)

            # Modern Prompt Template
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful academic assistant. Answer based on context: {context}"),
                ("human", "{input}"),
            ])
            
            # Combine into a modern chain
            doc_chain = create_stuff_documents_chain(llm, prompt)
            st.session_state.rag_chain = create_retrieval_chain(retriever, doc_chain)
            
            st.success("Analysis Complete!")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# --- CHAT ---
st.divider()
query = st.text_input("Ask a question:")
if query and "rag_chain" in st.session_state:
    response = st.session_state.rag_chain.invoke({"input": query})
    st.markdown(f"### 🤖 Answer:\n{response['answer']}")
