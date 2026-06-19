"""
tools/registry.py — manages all the tools the agent can use

each tool is just a function with a name, description, and input schema.
the agent reads the descriptions to decide which tool to call — so good
descriptions matter way more than you'd think.
"""

from dataclasses import dataclass
from typing import Callable, Any, Dict, List, Optional


@dataclass
class Tool:
    name: str
    description: str          # the agent reads this to know when to use the tool
    input_schema: Dict        # tells the agent what parameters to pass
    func: Callable            # the actual function that runs

    def run(self, **kwargs) -> str:
        """run the tool and always return a string — agents work with text"""
        try:
            result = self.func(**kwargs)
            return str(result)
        except Exception as e:
            # return the error as a string so the agent can see what went wrong
            # and potentially try a different approach
            return f"Tool error: {e}"


class ToolRegistry:
    """
    basically a phonebook for tools.
    agent looks up a tool by name, reads what it does, then calls it.
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        print(f"[Registry] Registered tool: {tool.name}")

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def get_all(self) -> List[Tool]:
        return list(self._tools.values())

    def format_for_prompt(self) -> str:
        """
        formats all tools into a string we inject into the agent's system prompt.
        the agent needs to know what tools exist and how to call them.
        """
        lines = []
        for tool in self._tools.values():
            # format each tool clearly so the LLM understands it
            params = ", ".join(
                f"{k}: {v}" for k, v in tool.input_schema.items()
            )
            lines.append(f"- {tool.name}({params}): {tool.description}")
        return "\n".join(lines)

    def names(self) -> List[str]:
        return list(self._tools.keys())
