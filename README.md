# Agentic RAG System

A complete Retrieval-Augmented Generation (RAG) system that enables intelligent question-answering from PDF documents. Built with FastAPI backend and Streamlit frontend, powered by local LLM inference via Ollama.

## 🚀 Features

- **FastAPI Backend**: Robust, asynchronous API server for AI processing
- **Streamlit Frontend**: Clean, interactive web interface
- **Local LLM**: Ollama-powered language models for privacy and cost efficiency
- **LangChain Agent**: ReAct framework for intelligent reasoning and tool usage
- **FAISS Vector Store**: High-performance vector database for document retrieval
- **PDF Knowledge Base**: Adaptable to any PDF document source
- **Modular Architecture**: Clean separation between frontend, backend, and agent logic

## 🏗️ Architecture

```
┌─────────────────┐    HTTP Requests    ┌─────────────────┐    LLM & Vector Store    ┌─────────────────┐
│                 │ ◄─────────────────► │                 │ ◄──────────────────────► │                 │
│   Streamlit     │                     │    FastAPI      │                          │   RAG Agent     │
│  (Frontend UI)  │                     │  (Backend API)  │                          │  (LangChain)    │
│                 │                     │                 │                          │                 │
└─────────────────┘                     └─────────────────┘                          └─────────────────┘
         │                                                                                     │
         │                                                                                     ▼
         │                                                                          ┌─────────────────────┐
         └─────────────────── User ───────────────────────────────────────────────► │ Ollama (LLM)       │
                                                                                    │ FAISS (Vector DB)  │
                                                                                    └─────────────────────┘
```

## 📋 Prerequisites

- **Python 3.10+**
- **Ollama**: Install from [ollama.com](https://ollama.com) and keep running

## 📁 Project Structure

```
rag/
├── backend/
│   ├── faiss_index_dataintensive/     # Auto-generated vector store
│   ├── main.py                        # FastAPI application
│   ├── agent_logic.py                 # LangChain logic
│   ├── requirements.txt               # Backend dependencies
│   └── Designing_data_intesive_applications.pdf  # Your PDF here
├── frontend/
│   ├── app.py                         # Streamlit application
│   └── requirements.txt               # Frontend dependencies
└── README.md                          # This file
```

## 🛠️ Installation & Setup

### Step 1: Global Setup

1. **Place Your PDF**: Put your PDF file in the `backend/` directory
   - Default: `Designing_data_intesive_applications.pdf`
   - To use a different file, update `PDF_PATH` in `backend/agent_logic.py`

2. **Install Ollama Model**:
   ```bash
   ollama pull deepseek-coder:6.7b-instruct
   ```
   - Ensure Ollama is running in the background

### Step 2: Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Step 3: Frontend Setup

1. **Navigate to frontend directory** (in a new terminal):
   ```bash
   cd frontend
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Running the Application

### Start Backend Server

In your backend terminal (with venv active):
```bash
python -m uvicorn main:app --reload
```

**Note**: First run will be slow as it processes the PDF and creates the FAISS index. Subsequent runs will be much faster.

### Start Frontend Application

In your frontend terminal (with venv active):
```bash
streamlit run app.py
```

Your browser should automatically open to the Streamlit application (usually at `http://localhost:8501`).

## 💡 Usage

1. Open the web interface
2. Type your question about the PDF content
3. Click "Get Answer"
4. View the agent's response based on the document

## 🔧 Troubleshooting

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError` | Ensure correct virtual environment is activated and dependencies are installed |
| Ollama Connection Error | Verify Ollama desktop application is running |
| Agent iteration limit | Rephrase question to be more specific, or increase `max_iterations` in `agent_logic.py` |

## 📚 Dependencies

### Backend
- FastAPI
- LangChain
- FAISS
- PyPDF2
- Sentence Transformers

### Frontend
- Streamlit
- Requests

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available under the [MIT License](LICENSE).