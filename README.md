---
title: Construction Rag
emoji: 📈
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: This is a RAG-based question-answering system specialized in
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Construction Domain RAG System

A Retrieval-Augmented Generation (RAG) system specialized for the construction industry, capable of answering queries about construction safety, project management, and regulations.

## System Architecture

```mermaid
graph TD
    A[Data Sources] --> B[Data Loading]
    B --> C[Text Processing]
    C --> D[Vector Store]
    D --> E[RAG Chain]
    E --> F[Query Interface]
    
    subgraph "Data Sources"
        A1[PDF Documents] --> A
        A2[Kaggle Datasets] --> A
    end
    
    subgraph "Data Processing"
        B1[PDF Loader] --> B
        B2[Kaggle Loader] --> B
        C1[Text Splitting] --> C
        C2[Chunk Creation] --> C
    end
    
    subgraph "RAG Components"
        D1[FAISS Index] --> D
        D2[Embeddings] --> D
        E1[LLM] --> E
        E2[Prompt Template] --> E
    end
```

## Core Components

### 1. Data Loading (`ConstructionDataLoader`)
- Handles multiple data sources:
  - PDF documents (construction codes, safety guidelines)
  - Kaggle datasets (project management data)
- Implements flexible loading mechanisms for each data type
- Manages temporary storage for downloaded datasets

### 2. RAG System (`ConstructionRAG`)

```mermaid
flowchart LR
    A[User Query] --> B[Retriever]
    B --> C[Vector Store]
    C --> D[Document Chain]
    D --> E[LLM]
    E --> F[Response]
    
    subgraph "Context Enhancement"
        C --> G[Top 5 Relevant Chunks]
        G --> D
    end
```

## Key Libraries Used

| Library | Purpose | Usage |
|---------|---------|-------|
| `langchain` | RAG Framework | Core RAG functionality, chains, and document processing |
| `langchain_openai` | OpenAI Integration | LLM and embeddings integration |
| `FAISS` | Vector Store | Efficient similarity search for document retrieval |
| `PyPDF` | PDF Processing | Loading and parsing PDF documents |
| `pandas` | Data Processing | Processing structured data from Kaggle |
| `kaggle` | Data Source | Accessing Kaggle datasets |
| `python-dotenv` | Configuration | Managing environment variables |
| `gradio` | UI Interface | Web interface for the RAG system |

## System Flow

1. **Initialization**
```mermaid
sequenceDiagram
    participant User
    participant System
    participant DataLoader
    participant VectorStore
    participant LLM
    
    User->>System: Initialize
    System->>DataLoader: Load Documents
    DataLoader->>DataLoader: Process PDFs
    DataLoader->>DataLoader: Load Kaggle Data
    DataLoader->>System: Return Documents
    System->>VectorStore: Create Index
    System->>LLM: Setup Chain
    System->>User: Ready for Queries
```

2. **Query Processing**
```mermaid
sequenceDiagram
    participant User
    participant System
    participant VectorStore
    participant LLM
    
    User->>System: Ask Question
    System->>VectorStore: Retrieve Relevant Docs
    VectorStore->>System: Return Context
    System->>LLM: Generate Response
    LLM->>System: Return Answer
    System->>User: Display Response
```

## Environment Setup

Required environment variables:
- `OPENAI_API_KEY`: OpenAI API authentication
- Kaggle credentials (for dataset access)

## Directory Structure
```
construction-rag/
├── main.py              # Core RAG implementation
├── app.py              # Gradio interface
├── requirements.txt     # Dependencies
├── construction_data/   # PDF documents
└── temp_kaggle_data/   # Temporary dataset storage
```

## Features
- Context-aware responses using RAG
- Multiple data source integration
- Structured response format
- Source attribution in answers
- Inference capabilities with transparency

## Query Examples
The system can handle queries about:
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
# What are the fall protection requirements for construction workers?
# What are the best practices for managing construction site safety?
# How have construction costs changed over recent years? - no answer!
# What are the most important factors to consider in construction project planning?
# What are common quality control measures in construction projects?
# What documentation is required for construction project compliance?
