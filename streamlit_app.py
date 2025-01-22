import streamlit as st
from main import ConstructionRAG
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

# Set OpenMP environment variable to avoid conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Move environment loading to the top
if not os.getenv("OPENAI_API_KEY"):
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not found in environment variables!")
        st.stop()

# Page configuration
st.set_page_config(
    page_title="Construction Domain Expert",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None

@st.cache_resource(show_spinner=True)
def initialize_rag():
    """Initialize RAG system with caching"""
    with st.spinner('Loading Construction Domain Expert...'):
        try:
            rag = ConstructionRAG()
            
            if os.path.exists(rag.vectorstore_path):
                try:
                    vectorstore = FAISS.load_local(
                        rag.vectorstore_path, 
                        rag.embeddings,
                        allow_dangerous_deserialization=True
                    )
                except Exception as e:
                    st.warning("Error loading existing vectorstore. Creating new one...")
                    documents = rag.load_all_documents(include_kaggle_datasets=["project_management"])
                    vectorstore = rag.create_or_load_vectorstore(documents, force_recreate=True)
            else:
                documents = rag.load_all_documents(include_kaggle_datasets=["project_management"])
                vectorstore = rag.create_or_load_vectorstore(documents, force_recreate=True)
            
            return rag.setup_rag_chain(vectorstore)
        except Exception as e:
            st.error(f"Error initializing RAG system: {str(e)}")
            raise

# Initialize the system
if not st.session_state.initialized:
    st.session_state.chain = initialize_rag()
    st.session_state.initialized = True

# Display title and description
st.title("Construction Domain Expert 🏗️")
st.markdown("""
This AI assistant can help answer questions about construction safety, 
project management, and regulations.
""")

# Example questions in sidebar
with st.sidebar:
    st.header("Example Questions")
    example_questions = [
        "What are the best practices for managing construction site safety?",
        "What are the minimum requirements for stairway construction?",
        "What are the fall protection requirements for workers?",
        "What are the common causes of project delays?",
        "How do material costs impact project budgets?"
    ]
    
    for question in example_questions:
        if st.button(question):
            st.session_state.messages.append({"role": "user", "content": question})
            response = st.session_state.chain.invoke({"input": question})
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about construction..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        response = st.session_state.chain.invoke({"input": prompt})
        st.markdown(response["answer"])
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]}) 