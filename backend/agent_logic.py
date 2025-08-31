import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- CHANGE 1: Import the correct 'ChatOllama' class ---
from langchain_ollama.chat_models import ChatOllama

from langchain.chains import RetrievalQA
from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
from langchain_core.prompts import ChatPromptTemplate

# --- Configuration ---
PDF_PATH = "Designing_data_intesive_applications.pdf"
VECTOR_STORE_PATH = "faiss_index_dataintensive"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3"


class RAGAgent:
    """A class to encapsulate the RAG agent's logic and state."""

    def __init__(self):
        print("Initializing RAG Agent...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vector_store = self._create_or_load_vector_store()

        # --- CHANGE 2: Instantiate 'ChatOllama' instead of 'OllamaLLM' ---
        self.llm = ChatOllama(model=OLLAMA_MODEL)

        self.agent_executor = self._create_agent_executor()
        print(f"RAG Agent initialized successfully with model '{OLLAMA_MODEL}'.")

    def _load_and_split_pdf(self):
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
        print("Creating RAG chain and agent executor...")
        retriever = self.vector_store.as_retriever()

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        @tool
        def ask_book(query: str) -> dict:
            """
            Answers questions about designing data-intensive applications by retrieving
            relevant information from the book. Use this for any technical question
            related to the book's content.
            """
            return qa_chain.invoke({"query": query})

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that answers questions using the provided tools."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        agent = create_tool_calling_agent(self.llm, [ask_book], prompt)

        return AgentExecutor(
            agent=agent,
            tools=[ask_book],
            verbose=True
        )

    def ask(self, query: str) -> str:
        """Asks a question to the agent and returns a formatted answer."""
        print(f"Received query for agent: {query}")
        try:
            response = self.agent_executor.invoke({"input": query})

            final_answer = response.get("output", "No answer found.")

            intermediate_steps = response.get("intermediate_steps", [])
            if intermediate_steps:
                tool_output = intermediate_steps[0][1]
                answer = tool_output.get("result", "")
                sources = tool_output.get("source_documents", [])

                final_answer = answer
                if sources:
                    # Correctly calculate page numbers (add 1) and sort them
                    source_pages = sorted(
                        list(set(doc.metadata.get('page', -1) + 1 for doc in sources if 'page' in doc.metadata)))
                    if source_pages:
                        final_answer += f"\n\n**Sources:** Pages {', '.join(map(str, source_pages))}"

            return final_answer

        except Exception as e:
            error_message = f"The agent failed to process your request. Error: {e}"
            print(error_message)
            return error_message


# Instantiate the agent globally.
rag_agent = RAGAgent()