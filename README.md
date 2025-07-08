# RAG Document Q&A Application

A modern web application that uses Retrieval-Augmented Generation (RAG) to answer questions based on uploaded documents. The application features a beautiful, responsive UI built with HTML/CSS/JavaScript and a Flask backend powered by LangChain.

## Features

- 📁 **Document Upload**: Support for PDF, TXT, DOCX, CSV, and other document formats
- 🧠 **RAG Pipeline**: Uses LangChain with FAISS vector store and HuggingFace embeddings
- 💬 **Interactive Q&A**: Ask questions about uploaded documents and get intelligent answers
- 📚 **Source Tracking**: View which documents were used to generate answers
- 🎨 **Modern UI**: Beautiful, responsive interface with drag-and-drop file upload
- ⚡ **Real-time Processing**: Fast document processing and question answering

## Prerequisites

- Python 3.8 or higher
- OpenAI API key (or OpenRouter API key)

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root with your API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### Option 1: Using the startup script (Recommended)
```bash
python run.py
```

### Option 2: Direct Flask execution
```bash
python app.py
```

### Access the Application
1. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

2. **Upload documents**:
   - Drag and drop files or click to browse
   - Supported formats: PDF, TXT, DOCX, CSV, DOC
   - Click "Process Documents" to create embeddings

3. **Ask questions**:
   - Type your question in the input field
   - Press Enter or click "Ask Question"
   - View the answer along with source documents

## Project Structure

```
RAG_Demo/
├── app.py                 # Flask backend application
├── run.py                 # Startup script
├── rag_app.py            # Original Streamlit app (kept for reference)
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html       # Main UI template
└── docs/               # Sample documents
```

## API Endpoints

- `GET /` - Serve the main HTML page
- `POST /process-documents` - Process uploaded documents and create embeddings
- `POST /ask-question` - Ask questions about processed documents

## Technical Details

### Backend (Flask)
- **Framework**: Flask with CORS support
- **RAG Pipeline**: LangChain with FAISS vector store
- **Embeddings**: HuggingFace sentence-transformers
- **LLM**: DeepSeek R1 via OpenRouter API

### Frontend (HTML/CSS/JavaScript)
- **UI**: Modern, responsive design with CSS Grid and Flexbox
- **Interactions**: Drag-and-drop file upload, real-time feedback
- **Styling**: Gradient backgrounds, glassmorphism effects, smooth animations

## Configuration

You can modify the following settings in `app.py`:

- **Chunk size**: Change `chunk_size=500` in the text splitter
- **Chunk overlap**: Change `chunk_overlap=50` in the text splitter
- **Retrieval count**: Change `k=3` in the retriever
- **LLM model**: Change the model name in the ChatOpenAI initialization
- **Temperature**: Adjust the temperature for more/less creative responses

## Troubleshooting

1. **API Key Issues**: Make sure your `.env` file contains the correct API key
2. **Port Conflicts**: If port 5000 is in use, change the port in `app.py`
3. **Memory Issues**: For large documents, consider reducing chunk size
4. **File Upload Errors**: Ensure files are in supported formats
5. **Dependencies**: If you encounter import errors, run `pip install -r requirements.txt`

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Create `.env` file with your API key
3. Run: `python run.py`
4. Open: http://localhost:5000
5. Upload documents and start asking questions!

## License

This project is open source and available under the MIT License. 