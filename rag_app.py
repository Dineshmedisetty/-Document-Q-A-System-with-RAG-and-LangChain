import os
import streamlit as st
import tempfile
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader,
    UnstructuredFileLoader, CSVLoader
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = "https://openrouter.ai/api/v1"

st.title("🧠 Document Q&A with RAG")
st.markdown("Upload documents to ask questions based on their content.")

def get_file_loader(file_path, file_type):
    """Get appropriate loader based on file type"""
    if file_type == "pdf":
        return PyPDFLoader(file_path)
    elif file_type == "txt":
        return TextLoader(file_path)
    elif file_type == "docx":
        return UnstructuredWordDocumentLoader(file_path)
    elif file_type == "csv":
        return CSVLoader(file_path)
    else:
        return UnstructuredFileLoader(file_path)

def process_uploaded_files(uploaded_files):
    """Process uploaded files and create vector store"""
    docs = []
    
    for uploaded_file in uploaded_files:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Get file extension
        file_extension = uploaded_file.name.lower().split(".")[-1]
        
        try:
            # Load document
            loader = get_file_loader(tmp_file_path, file_extension)
            file_docs = loader.load()
            
            # Add filename to metadata
            for doc in file_docs:
                doc.metadata["source"] = uploaded_file.name
            
            docs.extend(file_docs)
            st.success(f"✅ Successfully loaded: {uploaded_file.name}")
            
        except Exception as e:
            st.error(f"❌ Failed to load {uploaded_file.name}: {e}")
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
    
    if not docs:
        st.warning("No documents were successfully loaded.")
        return None
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs_split = text_splitter.split_documents(docs)
    
    # Generate embeddings
    with st.spinner("Creating embeddings..."):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs_split, embeddings)
    
    retriever = vectorstore.as_retriever(search_type="similarity", k=3)
    return retriever

# File upload section
st.subheader("📁 Upload Documents")
uploaded_files = st.file_uploader(
    "Choose files to upload",
    accept_multiple_files=True,
    type=['pdf', 'txt', 'docx', 'csv', 'doc'],
    help="Upload PDF, TXT, DOCX, CSV, or other document files"
)

# Initialize session state for retriever
if 'retriever' not in st.session_state:
    st.session_state.retriever = None

# Process uploaded files
if uploaded_files:
    if st.button("🔄 Process Documents"):
        st.session_state.retriever = process_uploaded_files(uploaded_files)
        if st.session_state.retriever:
            st.success(f"✅ Successfully processed {len(uploaded_files)} file(s)!")

# Only show Q&A section if documents are processed
if st.session_state.retriever is not None:
    st.subheader("💬 Ask Questions")
    
    # Initialize LLM
    llm = ChatOpenAI(
        model_name="deepseek/deepseek-r1-0528:free",
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base,
        temperature=0.7
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=st.session_state.retriever,
        return_source_documents=True
    )
    
    # Ask questions
    query = st.text_input("🔍 Ask a question about the uploaded documents:", "")
    
    if query:
        with st.spinner("Thinking..."):
            result = qa_chain(query)
        
        st.subheader("🧠 Answer")
        st.write(result["result"])
        
        st.subheader("📚 Sources")
        for doc in result["source_documents"]:
            source = doc.metadata.get("source", "Unknown")
            st.write(f"- {source}")
            
        # Optional: Show relevant text chunks
        with st.expander("📖 View relevant text chunks"):
            for i, doc in enumerate(result["source_documents"]):
                st.write(f"**Chunk {i+1} from {doc.metadata.get('source', 'Unknown')}:**")
                st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                st.write("---")

else:
    st.info("👆 Please upload and process documents first to start asking questions.")