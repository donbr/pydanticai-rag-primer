"""RAG example with FastAPI - exposing vector search through a web API.

Run pgvector with:
    mkdir postgres-data
    docker run --rm -e POSTGRES_PASSWORD=postgres \
        -p 54320:5432 \
        -v `pwd`/postgres-data:/var/lib/postgresql/data \
        pgvector/pgvector:pg17

First, make sure you've built the search database:
    uv run -m pydantic_ai_examples.rag_ingestion

Then start the API server:
    uvicorn pydantic_ai_examples.rag_api:app --reload

Access the API:
    - API docs: http://localhost:8000/docs 
    - Direct query: http://localhost:8000/ask?question=How+do+I+configure+logfire+to+work+with+FastAPI?
"""

from __future__ import annotations as _annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Dict, Any

import asyncpg
import fastapi
import logfire
import pydantic_core
from fastapi import Depends, Request
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing_extensions import AsyncGenerator

from pydantic_ai import RunContext
from pydantic_ai.agent import Agent

# Configure logging
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_asyncpg()

# Models for API
class Question(BaseModel):
    question: str


class Answer(BaseModel):
    answer: str


@dataclass
class Deps:
    openai: AsyncOpenAI
    pool: asyncpg.Pool


# Database connection management
@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    # Setup connections that will be used throughout the app lifecycle
    server_dsn, database = (
        'postgresql://postgres:postgres@localhost:54320',
        'pydantic_ai_rag',
    )
    
    app.state.pool = await asyncpg.create_pool(f'{server_dsn}/{database}')
    app.state.openai = AsyncOpenAI()
    logfire.instrument_openai(app.state.openai)
    
    yield
    
    # Cleanup on shutdown
    await app.state.pool.close()


# Create FastAPI app with lifespan
app = fastapi.FastAPI(lifespan=lifespan)

# Instrument the FastAPI app after it's created
logfire.instrument_fastapi(app)


# Dependency to get resources
async def get_deps(request: Request) -> Deps:
    return Deps(
        openai=request.app.state.openai,
        pool=request.app.state.pool
    )


# Create the agent
agent = Agent('openai:gpt-4o', deps_type=Deps, instrument=True)


@agent.tool
async def retrieve(context: RunContext[Deps], search_query: str) -> str:
    """Retrieve documentation sections based on a search query."""
    with logfire.span(
        'create embedding for {search_query=}', search_query=search_query
    ):
        embedding = await context.deps.openai.embeddings.create(
            input=search_query,
            model='text-embedding-3-small',
        )

    assert len(embedding.data) == 1, (
        f'Expected 1 embedding, got {len(embedding.data)}, doc query: {search_query!r}'
    )
    embedding = embedding.data[0].embedding
    embedding_json = pydantic_core.to_json(embedding).decode()
    rows = await context.deps.pool.fetch(
        'SELECT url, title, content FROM doc_sections ORDER BY embedding <-> $1 LIMIT 8',
        embedding_json,
    )
    return '\n\n'.join(
        f'# {row["title"]}\nDocumentation URL:{row["url"]}\n\n{row["content"]}\n'
        for row in rows
    )


# API endpoints
@app.get("/")
async def root():
    return {"message": "RAG API is running. Go to /docs for API documentation."}


@app.get("/ask", response_model=Answer)
async def ask_question_get(question: str, deps: Deps = Depends(get_deps)):
    """Ask a question using a GET request with a query parameter."""
    logfire.info('API received question via GET: "{question}"', question=question)
    answer = await agent.run(question, deps=deps)
    return Answer(answer=answer.data)


@app.post("/ask", response_model=Answer)
async def ask_question_post(question: Question, deps: Deps = Depends(get_deps)):
    """Ask a question using a POST request with a JSON body."""
    logfire.info('API received question via POST: "{question}"', question=question.question)
    answer = await agent.run(question.question, deps=deps)
    return Answer(answer=answer.data)


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}