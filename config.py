"""
config.py — all the settings in one place so I'm not hunting through files
"""

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent


@dataclass
class AgentConfig:
    # using llama locally via ollama — no API bill at the end of the month
    llm_model: str = "llama3.2"
    llm_base_url: str = "http://localhost:11434/v1"

    # how many times the agent can think-act-observe before giving up
    max_iterations: int = 10

    # if the agent is stuck in a loop, bail after this many repeated actions
    max_repeated_actions: int = 3

    # temperature for the LLM — low keeps it focused, high makes it creative
    temperature: float = 0.1


@dataclass
class MemoryConfig:
    # how many messages to keep in short-term memory before summarizing
    short_term_limit: int = 20

    # where to save long-term memory between sessions
    memory_path: Path = ROOT / "data" / "memory.json"


@dataclass
class ToolConfig:
    # max characters to return from web search so we don't blow the context window
    search_max_chars: int = 2000

    # max lines of output from code execution — nobody needs 10,000 lines of print statements
    code_output_limit: int = 50

    # timeout for code execution in seconds — don't let infinite loops run forever
    code_timeout: int = 10

    # max characters from wikipedia — articles can be insanely long
    wiki_max_chars: int = 3000


@dataclass
class AutoAgentConfig:
    agent: AgentConfig = field(default_factory=AgentConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)


cfg = AutoAgentConfig()
