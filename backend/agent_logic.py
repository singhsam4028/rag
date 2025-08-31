import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

# We need PromptTemplate to create our custom, smarter prompt
from langchain.agents import AgentExecutor, create_react_agent, tool
from langchain.prompts import PromptTemplate

# --- Configuration ---
PDF_PATH = "Designing_data_intesive_applications.pdf"
VECTOR_STORE_PATH = "faiss_index_dataintensive"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "deepseek-coder:6.7b-instruct"

# --- CHANGE 1: Define a more robust, custom prompt template ---
# This template explicitly tells the agent to provide a final answer
# once it has retrieved information from a tool.
REACT_TEMPLATE = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}
"""


class RAGAgent:
    """A class to encapsulate the RAG agent's logic and state."""

    def __init__(self):
        print("Initializing RAG Agent...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vector_store = self._create_or_load_vector_store()
        self.llm = OllamaLLM(model=OLLAMA_MODEL)
        self.agent_executor = self._create_agent_executor()
        print(f"RAG Agent initialized successfully with ReAct agent and model '{OLLAMA_MODEL}'.")

    def _load_and_split_pdf(self):
        # ... (This function is correct, no changes needed)
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
        # ... (This function is correct, no changes needed)
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
        """Creates the ReAct agent executor using our custom prompt."""
        print("Creating RAG chain and ReAct agent executor with custom prompt...")
        retriever = self.vector_store.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=retriever)

        @tool
        def ask_book(query: str) -> str:
            """
            Answers questions about designing data-intensive applications by retrieving
            relevant information from the book. Use this for any technical question
            related to the book's content. The input should be a clear, specific question.
            """
            return qa_chain.invoke({"query": query})["result"]

        tools = [ask_book]

        # --- CHANGE 2: Create the prompt from our custom template ---
        prompt = PromptTemplate.from_template(REACT_TEMPLATE)

        agent = create_react_agent(self.llm, tools, prompt)

        # --- CHANGE 3: Add a lower iteration limit as a safety measure ---
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5  # Set a reasonable limit to prevent runaway loops
        )

    def ask(self, query: str) -> str:
        """Asks a question to the agent and returns a formatted answer."""
        print(f"Received query for agent: {query}")
        try:
            response = self.agent_executor.invoke({"input": query})
            return response.get("output", "The agent did not return a final answer.")
        except Exception as e:
            # Handle the specific case of the iteration limit error for a better UX
            if "iteration limit" in str(e).lower():
                return "The agent took too many steps and was unable to find a final answer. This may be due to a complex query or model confusion. Please try rephrasing your question."
            error_message = f"The agent failed to process your request. Error: {e}"
            print(error_message)
            return error_message


# Instantiate the agent globally.
rag_agent = RAGAgent()