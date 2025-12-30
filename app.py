import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
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

# --- SIDEBAR: PDF UPLOAD ---
with st.sidebar:
    st.header("📂 Study Materials")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    process_button = st.button("Analyze Documents")

# --- CORE RAG LOGIC (2025 UPDATED) ---
if uploaded_files and process_button:
    with st.spinner("Analyzing documents..."):
        try:
            all_docs = []
            for uploaded_file in uploaded_files:
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                loader = PyPDFLoader(temp_path)
                all_docs.extend(loader.load())
                os.remove(temp_path)

            # 1. Split Text
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(all_docs)

            # 2. Generate Embeddings (v004 is the 2025 stable version)
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004", 
                google_api_key=api_key
            )

            # 3. Store in Memory (Stable for Streamlit Cloud)
            vectorstore = InMemoryVectorStore.from_documents(
                documents=splits, 
                embedding=embeddings
            )
            retriever = vectorstore.as_retriever()
            
            # 4. Initialize Modern Chain
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b", google_api_key=api_key)

            system_prompt = (
                "You are an academic research assistant. Use the following context to answer questions. "
                "If the answer isn't in the context, say you don't know based on the provided files.\n\n"
                "{context}"
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            # Combine everything into the RAG chain
            combine_docs_chain = create_stuff_documents_chain(llm, prompt)
            st.session_state.rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
            
            st.success("Analysis Complete! Ask questions below.")
            
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")

# --- CHAT INTERFACE ---
st.divider()
query = st.text_input("Ask a question about your study materials:")

if query:
    if "rag_chain" in st.session_state:
        with st.spinner("Finding answer..."):
            # New .invoke pattern for 2025
            response = st.session_state.rag_chain.invoke({"input": query})
            st.markdown(f"### 🤖 Analysis:\n{response['answer']}")
    else:
        st.warning("Please upload and analyze a PDF first.")

st.markdown("---")
st.caption("Powered by Gemini 1.5 & LangChain v1.x")
