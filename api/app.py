"""
api/app.py — FastAPI wrapper around the agent

exposes the agent as a REST API so it can be called from
a frontend, another service, or just curl.
"""

import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent.agent import ReActAgent
from agent.memory import AgentMemory
from tools.registry import ToolRegistry, Tool
from tools.web_search import web_search
from tools.wikipedia import wikipedia_search
from tools.calculator import calculate
from tools.code_executor import execute_python


# global agent instance — loaded once at startup
_agent: Optional[ReActAgent] = None


def build_agent() -> ReActAgent:
    """wire up all the tools and return a ready-to-use agent"""
    registry = ToolRegistry()

    registry.register(Tool(
        name="web_search",
        description="Search the web for current information, news, or facts not in Wikipedia",
        input_schema={"query": "string — what to search for"},
        func=web_search,
    ))

    registry.register(Tool(
        name="wikipedia",
        description="Look up detailed factual information about a topic on Wikipedia",
        input_schema={"topic": "string — the topic to look up"},
        func=wikipedia_search,
    ))

    registry.register(Tool(
        name="calculator",
        description="Evaluate math expressions: +, -, *, /, **, sqrt, log, sin, cos, pi",
        input_schema={"expression": "string — math expression to evaluate"},
        func=calculate,
    ))

    registry.register(Tool(
        name="execute_python",
        description="Write and run Python code to solve computational problems or process data",
        input_schema={"code": "string — valid Python code to execute"},
        func=execute_python,
    ))

    return ReActAgent(registry, AgentMemory())


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    print("[API] Loading AutoAgent...")
    _agent = build_agent()
    print("[API] Agent ready.")
    yield
    print("[API] Shutting down.")


app = FastAPI(
    title="AutoAgent API",
    description="ReAct-based autonomous AI agent with tool use and memory",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Request/Response models ────────────────────────────────────────────────────

class RunRequest(BaseModel):
    task: str = Field(..., min_length=3, max_length=2000, example="What is the capital of France?")
    verbose: bool = Field(False, description="Return the agent's full thought process")


class RunResponse(BaseModel):
    request_id: str
    task: str
    answer: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    tools_available: list
    version: str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        tools_available=_agent.tools.names() if _agent else [],
        version=app.version,
    )


@app.post("/run", response_model=RunResponse)
async def run_task(req: RunRequest):
    """
    give the agent a task and get back an answer.
    the agent will use whatever tools it needs to solve it.
    """
    if not _agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    t0 = time.perf_counter()
    answer = _agent.run(req.task, verbose=req.verbose)
    latency = (time.perf_counter() - t0) * 1000

    return RunResponse(
        request_id=str(uuid.uuid4()),
        task=req.task,
        answer=answer,
        latency_ms=round(latency, 2),
    )
