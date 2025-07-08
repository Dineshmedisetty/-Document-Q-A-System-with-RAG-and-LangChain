from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
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
import pickle

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = "https://openrouter.ai/api/v1"

app = Flask(__name__)
CORS(app)

# Global variable to store the retriever
retriever = None

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
    global retriever
    docs = []
    
    for uploaded_file in uploaded_files:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.filename.split('.')[-1]}") as tmp_file:
            uploaded_file.save(tmp_file.name)
            tmp_file_path = tmp_file.name
        
        # Get file extension
        file_extension = uploaded_file.filename.lower().split(".")[-1]
        
        try:
            # Load document
            loader = get_file_loader(tmp_file_path, file_extension)
            file_docs = loader.load()
            
            # Add filename to metadata
            for doc in file_docs:
                doc.metadata["source"] = uploaded_file.filename
            
            docs.extend(file_docs)
            print(f"✅ Successfully loaded: {uploaded_file.filename}")
            
        except Exception as e:
            print(f"❌ Failed to load {uploaded_file.filename}: {e}")
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
    
    if not docs:
        print("No documents were successfully loaded.")
        return None
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs_split = text_splitter.split_documents(docs)
    
    # Generate embeddings
    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs_split, embeddings)
    
    retriever = vectorstore.as_retriever(search_type="similarity", k=3)
    return retriever

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/process-documents', methods=['POST'])
def process_documents():
    """API endpoint to process uploaded documents"""
    global retriever
    
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if not files or all(file.filename == '' for file in files):
            return jsonify({'error': 'No files selected'}), 400
        
        # Process the files
        retriever = process_uploaded_files(files)
        
        if retriever is None:
            return jsonify({'error': 'Failed to process documents'}), 500
        
        return jsonify({'message': f'Successfully processed {len(files)} file(s)'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ask-question', methods=['POST'])
def ask_question():
    """API endpoint to ask questions about processed documents"""
    global retriever
    
    try:
        if retriever is None:
            return jsonify({'error': 'No documents processed. Please upload and process documents first.'}), 400
        
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'error': 'No question provided'}), 400
        
        question = data['question']
        
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
            retriever=retriever,
            return_source_documents=True
        )
        
        # Get answer
        result = qa_chain(question)
        
        # Extract sources and relevant text
        sources = [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]
        relevant_text = [doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content 
                        for doc in result["source_documents"]]
        
        return jsonify({
            'answer': result["result"],
            'sources': sources,
            'relevant_text': relevant_text
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000) 