import gradio as gr
from main import ConstructionRAG
import os
from dotenv import load_dotenv

# Load environment variables from .env file in development
load_dotenv()

# Get API key from environment variables (works with both .env and HF Spaces secrets)
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Initialize the RAG system
def initialize_rag():
    try:
        rag = ConstructionRAG()
        documents = rag.load_all_documents(include_kaggle_datasets=[
            "project_management",
        ])
        vectorstore = rag.create_vectorstore(documents)
        return rag.setup_rag_chain(vectorstore)
    except Exception as e:
        print(f"Error initializing RAG system: {str(e)}")
        raise

# Initialize the chain
chain = initialize_rag()

# Define the prediction function
def predict(message, history):
    response = chain.invoke({"input": message})
    return response["answer"]

# Define example questions
example_questions = [
    "What are the best practices for managing construction site safety?",
    "What are the minimum requirements for stairway construction in residential buildings?",
    "What are the fall protection requirements for construction workers?",
    "What are the key safety measures required when working at heights?",
    "What are the common causes of construction project delays?",
    "What is the average duration of residential construction projects?",
    "How do material costs typically impact construction project budgets?",
    "What are the most common risk factors in construction project management?",
    "What is the relationship between project size and completion time?",
    "What are the fall protection requirements for construction workers?",
    "What are the best practices for managing construction site safety?",
    "What are the most important factors to consider in construction project planning?",
    "What are common quality control measures in construction projects?",
    "What documentation is required for construction project compliance?"
]

# Create and launch the Gradio interface
demo = gr.ChatInterface(
    predict,
    chatbot=gr.Chatbot(height=600, type="messages"),
    textbox=gr.Textbox(
        placeholder="Ask me anything about construction...",
        container=False,
        scale=7
    ),
    title="Construction Domain Expert",
    description="I can help answer questions about construction safety, project management, and regulations.",
    theme="soft",
    examples=example_questions,
)

if __name__ == "__main__":
    demo.launch(share=True)