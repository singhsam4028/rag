import streamlit as st
import requests
import json

# --- Configuration ---
BACKEND_URL = "http://127.0.0.1:8000/ask"

# --- Streamlit App ---

# Set the title and a small description for the web app
st.set_page_config(page_title="Ask the Book", layout="wide")
st.title("📖 Ask the Book: Designing Data-Intensive Applications")
st.markdown("This is a simple web app that allows you to ask questions about the book 'Designing Data-Intensive Applications'. Your questions are answered by an AI agent with a knowledge base built from the book.")

# Input text box for the user's question
user_question = st.text_input("Enter your question:", "")

# Submit button
if st.button("Get Answer"):
    if user_question:
        # Show a spinner while waiting for the response
        with st.spinner("The agent is thinking..."):
            try:
                # The data to be sent to the API
                payload = {"query": user_question}

                # Making the POST request to the backend
                response = requests.post(BACKEND_URL, json=payload)
                response.raise_for_status()  # Raise an exception for bad status codes

                # Displaying the answer
                result = response.json()
                st.subheader("Agent's Answer:")
                st.write(result.get("answer", "No answer found."))

            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to the backend: {e}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question.")