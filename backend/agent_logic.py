import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.agents import AgentExecutor, create_react_agent, tool
from langchain import hub

# --- Configuration ---
PDF_PATH = "Designing_data_intesive_applications.pdf"
VECTOR_STORE_PATH = "faiss_index_dataintensive"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama2"

class RAGAgent:
    """A class to encapsulate the RAG agent's logic and state."""
    def __init__(self):
        """Initializes the agent, loading all necessary components."""
        print("Initializing RAG Agent...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vector_store = self._create_or_load_vector_store()
        self.llm = Ollama(model=OLLAMA_MODEL)
        self.agent_executor = self._create_agent_executor()
        print("RAG Agent initialized successfully.")

    def _load_and_split_pdf(self):
        """Loads and splits the PDF document."""
        if not os.path.exists(PDF_PATH):
            raise FileNotFoundError(f"PDF file not found at {PDF_PATH}")

        print(f"Loading and splitting document from {PDF_PATH}...")
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        print(f"Document split into {len(chunks)} chunks.")
        return chunks

    def _create_or_load_vector_store(self):
        """Creates or loads the FAISS vector store."""
        if os.path.exists(VECTOR_STORE_PATH):
            print(f"Loading existing vector store from {VECTOR_STORE_PATH}...")
            return FAISS.load_local(VECTOR_STORE_PATH, self.embeddings, allow_dangerous_deserialization=True)
        else:
            print("Creating new vector store...")
            chunks = self._load_and_split_pdf()
            vector_store = FAISS.from_documents(chunks, self.embeddings)
            vector_store.save_local(VECTOR_STORE_PATH)
            print(f"Vector store created and saved at {VECTOR_STORE_PATH}.")
            return vector_store

    def _create_agent_executor(self):
        """Creates the agent executor with a RAG chain tool."""
        print("Creating RAG chain and agent executor...")
        retriever = self.vector_store.as_retriever()

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )

        @tool
        def ask_book_on_data_intensive_applications(query: str) -> str:
            """
            Answers questions about designing data-intensive applications by retrieving
            relevant information from the provided book's content.
            """
            return qa_chain.invoke({"query": query})["result"]

        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(self.llm, [ask_book_on_data_intensive_applications], prompt)
        return AgentExecutor(agent=agent, tools=[ask_book_on_data_intensive_applications], verbose=True)

    def ask(self, query: str) -> str:
        """Asks a question to the agent and returns the answer."""
        print(f"Received query for agent: {query}")
        response = self.agent_executor.invoke({"input": query})
        return response["output"]

# Instantiate the agent globally. This ensures it's loaded only once at startup.
rag_agent = RAGAgent()