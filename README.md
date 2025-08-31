Agentic RAG System with FastAPI and Streamlit
This project implements a complete Retrieval-Augmented Generation (RAG) system that allows users to ask questions about a specific PDF document through a simple web interface. The backend is powered by a high-performance FastAPI server, and the frontend is a user-friendly Streamlit application.
The core of the system is a LangChain agent that uses a local Large Language Model (LLM) via Ollama and a FAISS vector store to provide accurate, context-aware answers based on the contents of the PDF.
Architecture
The application is designed with a clean client-server architecture:
+------------------+      (HTTP Request)      +-----------------+      (LLM & Vector Store)      +-----------------+
|                  |  <-------------------->  |                 |  <------------------------->  |                 |
| Streamlit        |                          | FastAPI         |                               | RAG Agent       |
| (Frontend UI)    |                          | (Backend API)   |                               | (LangChain)     |
|                  |                          |                 |                               |                 |
+------------------+                          +-----------------+                               +-----------------+
        ^                                                                                               |
        |                                                                                               v
        |                                                                                    +--------------------+
        +-------------------------------- User ---------------------------------------------+ Ollama (LLM)       |
                                                                                             + FAISS (Vector DB)  |
                                                                                             +--------------------+

Features
FastAPI Backend: A robust, asynchronous API server to handle all AI processing.
Streamlit Frontend: A clean, simple, and interactive web UI for user interaction.
Local LLM Inference: Uses Ollama to run powerful language models locally, ensuring privacy and no API costs.
LangChain Agent Framework: Utilizes a ReAct Agent for robust reasoning and tool use.
FAISS Vector Store: Creates and persists a high-speed vector database for efficient document retrieval.
PDF Knowledge Base: Easily adaptable to use any PDF document as the source of truth.
Modular Codebase: A clean separation of concerns between the frontend, backend, and agent logic.
Prerequisites
Before you begin, ensure you have the following installed on your system:
Python 3.10+
Ollama: You must have the Ollama desktop application installed and running. You can download it from ollama.com.
rag_web_app/
├── backend/
│   ├── faiss_index_dataintensive/  (Created automatically)
│   ├── main.py                     # FastAPI application
│   ├── agent_logic.py              # All LangChain logic
│   ├── requirements.txt
│   └── Designing_data_intesive_applications.pdf  <-- Place your PDF here
│
├── frontend/
│   ├── app.py                      # Streamlit application
│   └── requirements.txt
│
└── README.md                       # This file

Setup and Installation Guide
Follow these steps carefully to get the application running.
Part 1: Global Setup
Place Your PDF:
Put your PDF file inside the backend/ directory.
The project is currently configured to use a file named Designing_data_intesive_applications.pdf. If your file has a different name, you must update the PDF_PATH variable in backend/agent_logic.py.
Install the Correct Ollama Model:
The agent is optimized to work with an instruction-tuned model that is good at following the ReAct framework.
Open your terminal and run the following command to download the required model:
code
Bash
ollama pull deepseek-coder:6.7b-instruct
Ensure the Ollama application is running in the background.
Part 2: Backend Setup
You will need a terminal window for this section.
Navigate to the backend directory:
code
Bash
cd path/to/rag_web_app/backend
Create and activate a Python virtual environment:
code
Bash
python -m venv venv
source venv/bin/activate
(On Windows, use venv\Scripts\activate)
Install the required Python packages:
code
Bash
pip install -r requirements.txt
Part 3: Frontend Setup
You will need a second, separate terminal window for this section.
Navigate to the frontend directory:
code
Bash
cd path/to/rag_web_app/frontend
Create and activate another Python virtual environment:
code
Bash
python -m venv venv
source venv/bin/activate
(On Windows, use venv\Scripts\activate)
Install the required Python packages:
code
Bash
pip install -r requirements.txt
Running the Application
Now that setup is complete, you can start the servers.
Step 1: Start the Backend Server
In your backend terminal (with its venv active), run the following command:
code
Bash
python -m uvicorn main:app --reload
Important: The very first time you run this, it will be slow. The server needs to process the entire PDF, create embeddings, and save the FAISS vector store to the faiss_index_dataintensive folder.
On subsequent runs, it will be much faster as it will load the pre-built index.
Leave this terminal running.
Step 2: Start the Frontend Application
In your frontend terminal (with its venv active), run the following command:
code
Bash
streamlit run app.py
Your default web browser should automatically open with the Streamlit application running.
How to Use
Open the web page (usually at http://localhost:8501).
Type a question related to the content of your PDF into the text box.
Click the "Get Answer" button.
The agent's answer, derived from the book, will appear on the page.
Troubleshooting
ModuleNotFoundError: This error means a required package is missing. Make sure you have activated the correct virtual environment (source venv/bin/activate) in the correct directory (backend or frontend) before running the pip install -r requirements.txt command.
Ollama Connection Error: If the application can't connect to Ollama, ensure the Ollama desktop application is running on your machine.
Agent stopped due to iteration limit: This means the agent is stuck in a reasoning loop. This can happen with very complex or ambiguous questions. Try rephrasing your question to be more direct and specific. You can also increase the max_iterations value in backend/agent_logic.py as a last resort.