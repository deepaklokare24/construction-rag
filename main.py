import os
import sys
import kaggle
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from typing import List, Dict
import argparse

try:
    import faiss
    print(f"FAISS version: {faiss.__version__}")
except ImportError as e:
    print(f"Detailed import error: {e}")
    print(f"Python path: {sys.path}")

class ConstructionDataLoader:
    def __init__(self, pdf_directory: str):
        """Initialize the data loader with directory paths"""
        self.pdf_directory = pdf_directory
        self.kaggle_datasets = {
            "project_management": "claytonmiller/construction-and-project-management-example-data",
        }

    def load_pdf_documents(self) -> List[Document]:
        """Load documents from PDF files"""
        documents = []
        print(f"Looking for PDFs in: {self.pdf_directory}")
        pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]
        print(f"Found PDF files: {pdf_files}")
        
        for filename in pdf_files:
            file_path = os.path.join(self.pdf_directory, filename)
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
                print(f"Loaded PDF: {filename} with {len(docs)} pages")
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
        return documents

    def load_kaggle_dataset(self, dataset_name: str) -> List[Document]:
        """Load and process Kaggle dataset"""
        try:
            if dataset_name not in self.kaggle_datasets:
                raise ValueError(f"Unknown dataset: {dataset_name}")

            # Create a temporary directory for dataset download
            temp_dir = "temp_kaggle_data"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Download dataset using kaggle API
            kaggle.api.dataset_download_files(
                self.kaggle_datasets[dataset_name],
                path=temp_dir,
                unzip=True
            )
            
            # Process CSV files in the dataset
            documents = []
            for filename in os.listdir(temp_dir):
                if filename.endswith('.csv'):
                    file_path = os.path.join(temp_dir, filename)
                    df = pd.read_csv(file_path)
                    # Convert DataFrame to documents
                    for _, row in df.iterrows():
                        # Combine all columns into a single text
                        content = " ".join(f"{col}: {val}" for col, val in row.items())
                        doc = Document(
                            page_content=content,
                            metadata={"source": f"kaggle/{dataset_name}", "file": filename}
                        )
                        documents.append(doc)
            
            print(f"Loaded Kaggle dataset: {dataset_name}")
            return documents
        except Exception as e:
            print(f"Error loading Kaggle dataset {dataset_name}: {str(e)}")
            return []

class ConstructionRAG:
    def __init__(self):
        """Initialize the Construction RAG system"""
        load_dotenv()
        self.validate_environment()
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.embeddings = OpenAIEmbeddings()
        self.output_parser = StrOutputParser()
        self.data_loader = ConstructionDataLoader("construction_data")
        self.vectorstore_path = "vectorstore"  # Path to save/load vectorstore

    def validate_environment(self):
        """Validate required environment variables"""
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError(
                "Missing OPENAI_API_KEY in environment variables. "
                "Please set it in your .env file."
            )

    def load_all_documents(self, include_kaggle_datasets: List[str] = None):
        """Load all documents from PDFs and specified Kaggle datasets"""
        documents = self.data_loader.load_pdf_documents()
        print(f"Loaded {len(documents)} PDF documents")
        
        if include_kaggle_datasets:
            for dataset in include_kaggle_datasets:
                kaggle_docs = self.data_loader.load_kaggle_dataset(dataset)
                print(f"Loaded {len(kaggle_docs)} documents from {dataset}")
                documents.extend(kaggle_docs)

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        split_docs = text_splitter.split_documents(documents)
        print(f"Total chunks after splitting: {len(split_docs)}")
        return split_docs

    def create_or_load_vectorstore(self, documents=None, force_recreate=False):
        """Create new vectorstore or load existing one"""
        print(f"Creating new vector store with {len(documents)} documents")
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        
        # Save the vectorstore
        print("Saving vector store to disk...")
        vectorstore.save_local(self.vectorstore_path)
        print("Vector store saved successfully")
        
        return vectorstore

    def setup_rag_chain(self, vectorstore):
        """Set up the RAG chain for query processing"""
        prompt = ChatPromptTemplate.from_template("""
        You are a knowledgeable construction domain expert. Use the provided context to answer the question. 
        
        Guidelines:
        - If you find ANY relevant information in the context, use it to provide a partial answer
        - If you can make reasonable inferences from the available information, do so while noting your assumptions
        - Highlight what information you do have, even if it doesn't completely answer the question
        - If specific details are missing, mention what additional information would be helpful
        - If you truly have no relevant information, say "I don't have any relevant information about this topic in my current knowledge base."

        Context: {context}
        
        Question: {input}
        
        When citing sources, include:
        - PDF file names or Kaggle dataset names
        - Page numbers if available
        - Any relevant metadata
        
        Answer in a structured way:
        1. Direct information from sources (if any)
        2. Reasonable inferences (if applicable)
        3. Limitations or gaps in the available information
        """)

        document_chain = create_stuff_documents_chain(
            self.llm, 
            prompt,
            output_parser=self.output_parser
        )

        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5}  # Increased from 3 to 5 to get more context
        )
        return create_retrieval_chain(retriever, document_chain)

def main():
    try:
        # Initialize the RAG system
        rag = ConstructionRAG()
        
        # Create or load vectorstore
        print("Initializing vector store...")
        if os.path.exists(rag.vectorstore_path):
            print("Found existing vector store, loading...")
            vectorstore = FAISS.load_local(
                rag.vectorstore_path, 
                rag.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            print("No existing vector store found. Creating new one...")
            documents = rag.load_all_documents(include_kaggle_datasets=["project_management"])
            vectorstore = rag.create_or_load_vectorstore(documents, force_recreate=True)
        
        # Setup RAG chain
        print("Setting up RAG chain...")
        chain = rag.setup_rag_chain(vectorstore)
        
        # Interactive query loop
        print("\nConstruction Domain QA System Ready!")
        print("Type 'quit' to exit")
        
        while True:
            question = input("\nEnter your construction-related question: ")
            if question.lower() == 'quit':
                break
                
            response = chain.invoke({"input": question})
            print(f"\nAnswer: {response['answer']}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
    
    
# Tesitng
# What are the best practices for managing construction site safety?
# What are the minimum requirements for stairway construction in residential buildings?
# What are the fall protection requirements for construction workers?
# What are the emergency exit requirements for residential buildings? - no answer!
# What are the key safety measures required when working at heights?
# What are the common causes of construction project delays?
# What is the average duration of residential construction projects?
# How do material costs typically impact construction project budgets? - Inference
# What are the most common risk factors in construction project management?
# What is the relationship between project size and completion time?
# How have construction costs changed over recent years? - no answer!
# What are the most important factors to consider in construction project planning?
# What are common quality control measures in construction projects?
# What documentation is required for construction project compliance?
