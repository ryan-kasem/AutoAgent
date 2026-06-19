"""
main.py — demo runner for AutoAgent

shows the agent solving different types of problems using its tools.
run this to see the full thought process printed out.

usage:
    python main.py                              # run all demos
    python main.py --task "your question here"  # ask a single question
    python main.py --benchmark                  # run evaluation suite
"""

import argparse
from tools.registry import ToolRegistry, Tool
from tools.web_search import web_search
from tools.wikipedia import wikipedia_search
from tools.calculator import calculate
from tools.code_executor import execute_python
from agent.agent import ReActAgent
from agent.memory import AgentMemory
from evaluation.benchmark import run_benchmark


def build_agent() -> ReActAgent:
    """set up all tools and return a ready agent"""
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


# demo tasks that show off different tool combinations
DEMO_TASKS = [
    "What is 15% of 2847, rounded to the nearest dollar?",
    "Who created the FAISS library and what is it used for?",
    "Write Python code to generate the first 10 Fibonacci numbers and return them as a list.",
    "What is the square root of 98596 and is that number prime? Show your work.",
]


def run_demos(agent: ReActAgent) -> None:
    print("\n" + "=" * 60)
    print("           AutoAgent Demo")
    print("=" * 60)

    for task in DEMO_TASKS:
        answer = agent.run(task, verbose=True)
        print(f"\nFINAL ANSWER: {answer}")
        print("\n" + "-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoAgent demo")
    parser.add_argument("--task", type=str, help="Run a single custom task")
    parser.add_argument("--benchmark", action="store_true", help="Run evaluation benchmark")
    args = parser.parse_args()

    print("[AutoAgent] Initializing...")
    agent = build_agent()

    if args.task:
        answer = agent.run(args.task, verbose=True)
        print(f"\nAnswer: {answer}")
    elif args.benchmark:
        report = run_benchmark(agent, verbose=False)
        report.print_summary()
    else:
        run_demos(agent)
        report = run_benchmark(agent, verbose=False)
        report.print_summary()
