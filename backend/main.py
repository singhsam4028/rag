from fastapi import FastAPI
from pydantic import BaseModel
from agent_logic import rag_agent

# Initialize the FastAPI app
app = FastAPI(
    title="Agentic RAG System API",
    description="An API for querying a book on data-intensive applications.",
    version="1.0.0"
)

# Define the request model for the question
class QuestionRequest(BaseModel):
    query: str

# Define the root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Agentic RAG API. Use the /ask endpoint to post your questions."}

# Define the main endpoint for asking questions
@app.post("/ask")
def ask_question(request: QuestionRequest):
    """
    Receives a question, processes it with the RAG agent, and returns the answer.
    """
    answer = rag_agent.ask(request.query)
    return {"answer": answer}