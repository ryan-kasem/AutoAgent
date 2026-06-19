# AutoAgent

An autonomous AI agent built on the ReAct (Reasoning + Acting) framework. Give it a task - it figures out what tools to use, calls them, observes the results, and keeps going until it has an answer.

---

## What it does

Instead of answering from memory like a chatbot, AutoAgent actually *executes* things:

- Runs real Python code in a sandboxed environment
- Searches the web for current information
- Looks up Wikipedia articles
- Evaluates math expressions with full precision

The agent decides which tool to use based on its reasoning, not hardcoded rules.

---

## How it works (ReAct loop)

Every task runs through this loop until the agent has an answer or hits the iteration limit:

```
Thought:       "I need to calculate this - I'll use the calculator tool"
Action:        calculator
Action Input:  {"expression": "sqrt(1764)"}
Observation:   The result is 42.
Thought:       "I have the answer"
Final Answer:  42
```

The LLM reasons out loud before every action. This makes the agent far more reliable than just directly calling tools - it catches its own mistakes mid-loop.

---

## Architecture

```
Task Input
    │
    ▼
System Prompt (tool descriptions injected)
    │
    ▼
ReAct Loop:
  LLM generates Thought + Action
    │
    ▼
  Tool Registry routes to the right tool
    │
    ▼
  Observation fed back to LLM
    │
    ▼
  Repeat until "Final Answer"
    │
    ▼
Answer + stored in long-term memory
```

---

## Tools

| Tool | Source | Use case |
|---|---|---|
| `web_search` | DuckDuckGo Instant Answer API | Current events, news, recent facts |
| `wikipedia` | Wikipedia REST API | Background knowledge, definitions |
| `calculator` | Safe `eval` with math namespace | Any arithmetic or math |
| `execute_python` | Sandboxed `exec()` with SIGALRM timeout | Complex computation, data processing |

Adding a new tool is ~10 lines - just register a `Tool` with a name, description, and function.

---

## Stack

| Component | Technology |
|---|---|
| LLM | Llama 3.2 via Ollama |
| Agent framework | Custom ReAct implementation |
| API | FastAPI + Uvicorn |
| Memory | In-memory (short-term) + JSON file (long-term) |
| Evaluation | Custom benchmark with tool-call tracking |

---

## Quick Start

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.ai) installed

```bash
# 1. Clone
git clone https://github.com/ryan-kasem/AutoAgent.git
cd AutoAgent

# 2. Install deps
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. Pull the LLM (free, runs locally)
ollama pull llama3.2
ollama serve  # run in a separate terminal tab

# 4. Run demo tasks
python main.py

# Run a custom task
python main.py --task "What is 15% of 2847?"

# Run the benchmark
python main.py --benchmark
```

---

## Project Structure

```
AutoAgent/
├── config.py                    # agent settings, model config
├── agent/
│   ├── agent.py                 # ReAct loop, action parsing
│   └── memory.py                # short-term + long-term memory
├── tools/
│   ├── registry.py              # tool registration + prompt formatting
│   ├── web_search.py            # DuckDuckGo search
│   ├── wikipedia.py             # Wikipedia REST API
│   ├── calculator.py            # safe math evaluator
│   └── code_executor.py        # sandboxed Python runner
├── evaluation/
│   └── benchmark.py             # benchmark tasks + report
├── api/
│   └── app.py                   # FastAPI REST API
└── main.py                      # demo runner
```

---

## Benchmark

Evaluated across 5 task types - math, factual lookup, code execution, and multi-step reasoning:

| Metric | Description |
|---|---|
| Success Rate | Answer contained expected keywords |
| Tool Accuracy | Agent used the correct tool category |
| Avg Iterations | How many steps to reach the answer |
| Avg Latency | End-to-end wall-clock time |

---

## References

- Yao et al. (2022) - [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
