"""
agent/memory.py — short-term and long-term memory for the agent

short-term: the current conversation / thought chain (lost when session ends)
long-term: facts the agent has learned, saved to disk between sessions

without memory the agent forgets everything between turns.
with memory it can say "oh I already looked this up earlier."
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
from config import cfg


@dataclass
class Message:
    role: str    # "user", "assistant", "tool", or "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ShortTermMemory:
    """
    stores the current conversation — basically the message history
    we pass to the LLM on every call.

    when it gets too long we summarize old messages to save tokens.
    this is the same trick ChatGPT uses when conversations get long.
    """

    def __init__(self, limit: int = None):
        self.limit = limit or cfg.memory.short_term_limit
        self.messages: List[Message] = []

    def add(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))
        # trim if we're over the limit — keep the most recent messages
        # always keep the first message (system prompt) though
        if len(self.messages) > self.limit:
            system_msgs = [m for m in self.messages if m.role == "system"]
            recent = self.messages[-(self.limit - len(system_msgs)):]
            self.messages = system_msgs + recent

    def to_llm_format(self) -> List[Dict[str, str]]:
        """convert to the format OpenAI/Ollama expects"""
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def clear(self) -> None:
        # keep system messages, clear everything else
        self.messages = [m for m in self.messages if m.role == "system"]

    def last_n(self, n: int) -> List[Message]:
        return self.messages[-n:]


class LongTermMemory:
    """
    persists facts between sessions by saving to a JSON file.
    the agent can explicitly store something it wants to remember
    and retrieve it later.

    not a vector database — just a simple key-value store.
    for a production system you'd use a proper vector DB here.
    """

    def __init__(self, path: Path = None):
        self.path = path or cfg.memory.memory_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._store: Dict[str, str] = self._load()

    def store(self, key: str, value: str) -> None:
        """save a fact — key is a short label, value is the content"""
        self._store[key] = value
        self._save()

    def retrieve(self, key: str) -> Optional[str]:
        return self._store.get(key)

    def search(self, query: str) -> List[str]:
        """
        naive keyword search over stored facts.
        good enough for a demo — real systems would use embeddings here.
        """
        query_lower = query.lower()
        matches = []
        for key, value in self._store.items():
            if query_lower in key.lower() or query_lower in value.lower():
                matches.append(f"{key}: {value}")
        return matches

    def all_keys(self) -> List[str]:
        return list(self._store.keys())

    def _load(self) -> Dict[str, str]:
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return {}

    def _save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self._store, f, indent=2)


class AgentMemory:
    """combines both memory types into one interface"""

    def __init__(self):
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()

    def remember(self, key: str, value: str) -> None:
        """explicitly store something in long-term memory"""
        self.long_term.store(key, value)

    def recall(self, query: str) -> str:
        """search long-term memory for relevant facts"""
        results = self.long_term.search(query)
        if not results:
            return "Nothing relevant found in memory."
        return "\n".join(results)
