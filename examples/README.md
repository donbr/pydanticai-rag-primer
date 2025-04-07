# PydanticAI RAG Examples

This directory contains the implementation code for the RAG system described in the main repository README.

## Examples Overview

The implementation consists of three main Python files:

1. **`rag_ingestion.py`**: Builds the vector database by processing documents and generating embeddings
2. **`rag_inference.py`**: Core component for answering questions using retrieved context
3. **`rag_api.py`**: FastAPI web interface for accessing the RAG system

## Example Usage

Each example file can be run independently:

```bash
# Build the database
python rag_ingestion.py

# Ask a question
python rag_inference.py "How do I configure logfire to work with FastAPI?"

# Start the API server
uvicorn rag_api:app --reload
```

### API Usage with curl

Once the API server is running, you can interact with it using curl:

```bash
# GET request
curl "http://localhost:8000/ask?question=How+do+I+configure+logfire+to+work+with+FastAPI%3F"

# POST request
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I configure logfire to work with FastAPI?"}'

# Health check
curl "http://localhost:8000/health"
```

## Implementation Details

These examples demonstrate key PydanticAI concepts:

- **Type-safe dependency injection** with the `@dataclass` pattern
- **Tool registration** using `@agent.tool` decorators
- **RunContext** for managing state and dependencies
- **Asynchronous programming** with asyncio and semaphores

## Learning from the Examples

The best way to understand these examples:

1. Start with `rag_inference.py` to see the core RAG pattern
2. Explore `rag_ingestion.py` to understand document processing
3. Review `rag_api.py` to learn about API integration

You can also validate and explore your PostgreSQL database using the `rag_postgres_validation.ipynb` Jupyter notebook.

For comprehensive installation and usage instructions, see the [main README](../README.md).