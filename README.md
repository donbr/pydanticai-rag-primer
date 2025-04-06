# PydanticAI RAG Primer

This repository contains a comprehensive implementation of a Retrieval Augmented Generation (RAG) system using PydanticAI. It demonstrates how to build a production-ready question-answering system that grounds LLM responses in external knowledge.

## 🔍 Overview

The RAG system enhances Large Language Model (LLM) capabilities by:
- Retrieving relevant information from a knowledge base
- Augmenting prompts with retrieved context
- Generating accurate, knowledge-grounded responses

This project provides a complete, modular implementation with:
- Document ingestion pipeline
- Vector similarity search
- Type-safe API integration
- PydanticAI agent architecture

## 📊 Architecture

![RAG Architecture](docs/pydanticai-rag-diagram.md)

The system is organized into several interconnected components:

- **Ingestion Pipeline**: Processes documents and generates embeddings
- **Vector Database**: Stores and indexes document embeddings for similarity search
- **PydanticAI Agent**: Provides type-safe framework for LLM interaction
- **Query Processing**: Handles user questions and retrieves relevant context
- **API Layer**: Exposes functionality through a web interface

## 🛠️ Components

The implementation consists of four main Python files:

1. **`rag_ingestion.py`**: Builds the vector database by processing documents and generating embeddings
2. **`rag_inference.py`**: Core component for answering questions using retrieved context
3. **`rag.py`**: Complete implementation with CLI for both ingestion and querying
4. **`rag_api.py`**: FastAPI web interface for accessing the RAG system

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or later
- Docker (for running PostgreSQL with pgvector)
- OpenAI API key

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/pydanticai-rag-primer.git
   cd pydanticai-rag-primer
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   ```

3. Set up the PostgreSQL database with pgvector:
   ```bash
   mkdir postgres-data
   docker run --rm -e POSTGRES_PASSWORD=postgres \
       -p 54320:5432 \
       -v $(pwd)/postgres-data:/var/lib/postgresql/data \
       pgvector/pgvector:pg17
   ```

4. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

### Usage

#### 1. Build the vector database

```bash
uv run -m pydantic_ai_examples.rag_ingestion
# or
uv run -m pydantic_ai_examples.rag build
```

#### 2. Ask questions via CLI

```bash
uv run -m pydantic_ai_examples.rag_inference "How do I configure logfire to work with FastAPI?"
# or
uv run -m pydantic_ai_examples.rag search "How do I configure logfire to work with FastAPI?"
```

#### 3. Run the web API

```bash
uvicorn pydantic_ai_examples.rag_api:app --reload
```

Then visit:
- API documentation: http://localhost:8000/docs
- Ask a question: http://localhost:8000/ask?question=How+do+I+configure+logfire+to+work+with+FastAPI?

## 💡 Key Features

- **Type-safe dependency injection** using PydanticAI
- **Efficient vector similarity search** with HNSW indexing
- **Concurrent processing** with asyncio for better performance
- **Clean separation of concerns** across modular components
- **Web API integration** with FastAPI
- **Comprehensive logging** with logfire

## 📚 Technical Details

### Vector Database

The system uses PostgreSQL with the pgvector extension for efficient vector similarity search:
- Document embeddings are created using OpenAI's `text-embedding-3-small` model
- HNSW (Hierarchical Navigable Small World) indexing for fast approximate nearest neighbor search
- Vector similarity queries using the `<->` operator

### PydanticAI Integration

The implementation leverages PydanticAI for type-safe agent development:
- Defines dependencies through a typed `Deps` class
- Registers a `retrieve` tool for vector search
- Uses `RunContext` to manage state and dependencies
- Configures the agent with the GPT-4o model

## 🔄 Extension Points

The RAG system can be extended in several ways:
1. Add more specialized tools to the agent
2. Implement different embedding models
3. Enhance retrieval with reranking or hybrid search
4. Add authentication to the API
5. Define structured output types for specific use cases

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [PydanticAI](https://github.com/pydantic/pydantic-ai) for the agent framework
- [pgvector](https://github.com/pgvector/pgvector) for vector similarity search in PostgreSQL
- [logfire](https://logfire.dev/) for structured logging