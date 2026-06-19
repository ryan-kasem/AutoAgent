"""
evaluation/benchmark.py — measures how well the agent actually works

we test the agent on a set of tasks with known answers and measure:
- success rate (did it get the right answer?)
- tool usage (did it use the right tools?)
- iteration efficiency (how many steps did it take?)

this is the kind of eval that makes a project look serious on a resume.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional
from agent.agent import ReActAgent


@dataclass
class BenchmarkTask:
    task: str
    expected_keywords: List[str]   # answer should contain these words
    expected_tools: List[str]      # agent should use these tools
    max_allowed_iterations: int = 8


@dataclass
class TaskResult:
    task: str
    answer: str
    success: bool
    tools_used: List[str]
    correct_tools: bool
    iterations: int
    latency_seconds: float
    notes: str = ""


@dataclass
class BenchmarkReport:
    results: List[TaskResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.success for r in self.results) / len(self.results)

    @property
    def avg_iterations(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.iterations for r in self.results) / len(self.results)

    @property
    def avg_latency(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.latency_seconds for r in self.results) / len(self.results)

    @property
    def tool_accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.correct_tools for r in self.results) / len(self.results)

    def print_summary(self) -> None:
        print("\n" + "=" * 55)
        print("        AutoAgent Benchmark Report")
        print("=" * 55)
        print(f"  Tasks evaluated:      {len(self.results)}")
        print(f"  Success rate:         {self.success_rate:.1%}")
        print(f"  Tool accuracy:        {self.tool_accuracy:.1%}")
        print(f"  Avg iterations:       {self.avg_iterations:.1f}")
        print(f"  Avg latency:          {self.avg_latency:.1f}s")
        print("-" * 55)
        for r in self.results:
            status = "PASS" if r.success else "FAIL"
            print(f"  [{status}] {r.task[:45]:<45} ({r.iterations} steps, {r.latency_seconds:.1f}s)")
            if not r.success:
                print(f"         Answer: {r.answer[:80]}...")
        print("=" * 55 + "\n")


# standard benchmark tasks — covers all major tool types
DEFAULT_TASKS = [
    BenchmarkTask(
        task="What is the square root of 1764?",
        expected_keywords=["42"],
        expected_tools=["calculator"],
    ),
    BenchmarkTask(
        task="Who invented the transformer architecture in machine learning and what year?",
        expected_keywords=["vaswani", "2017", "attention"],
        expected_tools=["wikipedia", "web_search"],
    ),
    BenchmarkTask(
        task="Write Python code to find all prime numbers up to 50 and tell me how many there are.",
        expected_keywords=["15", "prime"],
        expected_tools=["execute_python"],
    ),
    BenchmarkTask(
        task="What does FAISS stand for and who made it?",
        expected_keywords=["facebook", "similarity", "search"],
        expected_tools=["wikipedia", "web_search"],
    ),
    BenchmarkTask(
        task="Calculate (2^10 + 3^5) / 7 rounded to 2 decimal places.",
        expected_keywords=["149", "150"],
        expected_tools=["calculator"],
    ),
]


def run_benchmark(agent: ReActAgent, tasks: List[BenchmarkTask] = None, verbose: bool = False) -> BenchmarkReport:
    """
    run all benchmark tasks through the agent and return a report.
    set verbose=True to see the agent's full thought process for each task.
    """
    tasks = tasks or DEFAULT_TASKS
    report = BenchmarkReport()

    print(f"\n[Benchmark] Running {len(tasks)} tasks...")

    for i, task in enumerate(tasks):
        print(f"  Task {i+1}/{len(tasks)}: {task.task[:55]}...")

        # track which tools get called during this task
        tools_called = []
        original_run = {}

        # monkey-patch each tool to track usage — hacky but effective
        for tool_name in agent.tools.names():
            tool = agent.tools.get(tool_name)
            original_func = tool.func

            def make_tracked(name, fn):
                def tracked(**kwargs):
                    tools_called.append(name)
                    return fn(**kwargs)
                return tracked

            tool.func = make_tracked(tool_name, original_func)
            original_run[tool_name] = original_func

        # time the task
        t0 = time.time()
        try:
            answer = agent.run(task.task, verbose=verbose)
        except Exception as e:
            answer = f"Agent crashed: {e}"
        latency = time.time() - t0

        # restore original functions
        for tool_name, fn in original_run.items():
            agent.tools.get(tool_name).func = fn

        # check if the answer contains the expected keywords
        answer_lower = answer.lower()
        success = any(kw.lower() in answer_lower for kw in task.expected_keywords)

        # check if it used appropriate tools
        correct_tools = any(t in tools_called for t in task.expected_tools)

        # count iterations by looking at short-term memory length
        iterations = len([
            m for m in agent.memory.short_term.messages
            if m.role == "assistant"
        ])

        report.results.append(TaskResult(
            task=task.task,
            answer=answer,
            success=success,
            tools_used=tools_called,
            correct_tools=correct_tools,
            iterations=max(iterations, 1),
            latency_seconds=round(latency, 2),
        ))

    return report
