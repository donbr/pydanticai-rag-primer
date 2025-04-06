# PydanticAI RAG Examples

This directory contains the implementation code for the RAG system described in the main repository README.

## Examples Overview

The implementation consists of four main Python files:

1. **`rag_ingestion.py`**: Builds the vector database by processing documents and generating embeddings
2. **`rag_inference.py`**: Core component for answering questions using retrieved context
3. **`rag.py`**: Complete implementation with CLI for both ingestion and querying
4. **`rag_api.py`**: FastAPI web interface for accessing the RAG system

## Example Usage

Each example file can be run independently:

```bash
# Build the database
uv run -m pydantic_ai_examples.rag_ingestion

# Ask a question
uv run -m pydantic_ai_examples.rag_inference "How do I configure logfire to work with FastAPI?"

# Use the combined CLI
uv run -m pydantic_ai_examples.rag build  # Build the database
uv run -m pydantic_ai_examples.rag search "How do I configure logfire to work with FastAPI?"  # Search

# Start the API server
uvicorn pydantic_ai_examples.rag_api:app --reload
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
3. Study `rag.py` to see how both are combined with a CLI
4. Review `rag_api.py` to learn about API integration

For comprehensive installation and usage instructions, see the [main README](../README.md).