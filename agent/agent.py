"""
agent/agent.py — the ReAct agent loop

ReAct = Reasoning + Acting (Yao et al., 2022)
the core idea: instead of answering immediately, the agent:
  1. THINKS about what it needs to do (Reasoning)
  2. ACTS by calling a tool (Acting)
  3. OBSERVES the result
  4. repeats until it has enough info to answer

this loop is what separates an "agent" from a plain chatbot.
a chatbot just responds. an agent takes actions to find the answer.
"""

import json
import re
from typing import Optional
from openai import OpenAI

from agent.memory import AgentMemory
from tools.registry import ToolRegistry
from config import cfg


# the system prompt is basically the agent's brain.
# it tells the LLM exactly how to think and what format to use.
SYSTEM_PROMPT = """You are AutoAgent, an autonomous AI assistant that solves problems step by step using tools.

For every task, follow this loop:

Thought: think about what you need to do and what tool would help
Action: tool_name
Action Input: {{"param": "value"}}
Observation: (the tool result will appear here)
... (repeat as needed)
Thought: I now have enough information to answer
Final Answer: your complete answer here

Available tools:
{tools}

Rules:
- Always start with a Thought
- Use one tool at a time
- If a tool fails, try a different approach
- When you have the answer, say "Final Answer:" and stop
- Never make up information — use tools to find facts
- Be concise in your thoughts, detailed in your final answer
"""


class ReActAgent:
    """
    the main agent class.
    give it a task, it figures out how to solve it using whatever tools it has.
    """

    def __init__(self, tool_registry: ToolRegistry, memory: AgentMemory = None):
        self.tools = tool_registry
        self.memory = memory or AgentMemory()

        # connect to local ollama — same setup as NeuralRAG
        self.llm = OpenAI(
            base_url=cfg.agent.llm_base_url,
            api_key="ollama",
        )

    def run(self, task: str, verbose: bool = True) -> str:
        """
        run the agent on a task and return the final answer.
        verbose=True prints the thinking process — useful for debugging and demos.
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Task: {task}")
            print('='*60)

        # set up the system prompt with the available tools
        system = SYSTEM_PROMPT.format(tools=self.tools.format_for_prompt())

        # start fresh for each task
        self.memory.short_term.clear()
        self.memory.short_term.add("system", system)

        # check if we have any relevant memories from past sessions
        past_memories = self.memory.recall(task)
        if past_memories != "Nothing relevant found in memory.":
            self.memory.short_term.add(
                "system",
                f"Relevant information from past sessions:\n{past_memories}"
            )

        # add the user's task
        self.memory.short_term.add("user", task)

        # track repeated actions so we can bail if the agent gets stuck
        action_history = []

        for iteration in range(cfg.agent.max_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1} ---")

            # ask the LLM what to do next
            response = self._call_llm()

            if verbose:
                print(response)

            # did it reach a final answer?
            if "Final Answer:" in response:
                answer = response.split("Final Answer:")[-1].strip()
                # store successful task-answer pairs in long-term memory
                self.memory.remember(task[:50], answer[:200])
                return answer

            # try to parse the action the agent wants to take
            action, action_input = self._parse_action(response)

            if not action:
                # the agent didn't follow the format — nudge it
                self.memory.short_term.add("assistant", response)
                self.memory.short_term.add(
                    "user",
                    "Please continue. Remember to use the format: Action: tool_name"
                )
                continue

            # detect if the agent is stuck in a loop
            action_key = f"{action}:{json.dumps(action_input, sort_keys=True)}"
            action_history.append(action_key)
            repeated = sum(1 for a in action_history if a == action_key)
            if repeated >= cfg.agent.max_repeated_actions:
                stuck_msg = f"You've called {action} with the same input {repeated} times. Try a different approach."
                self.memory.short_term.add("assistant", response)
                self.memory.short_term.add("user", f"Observation: {stuck_msg}")
                continue

            # run the tool
            tool = self.tools.get(action)
            if not tool:
                observation = f"Unknown tool '{action}'. Available tools: {', '.join(self.tools.names())}"
            else:
                if verbose:
                    print(f"\n[Running tool: {action}({action_input})]")
                observation = tool.run(**action_input)
                if verbose:
                    print(f"[Observation]: {observation[:200]}...")

            # add the agent's reasoning + tool result to memory
            self.memory.short_term.add("assistant", response)
            self.memory.short_term.add("user", f"Observation: {observation}")

        # hit max iterations without a final answer
        return "I wasn't able to complete this task within the iteration limit. Please try rephrasing or breaking it into smaller steps."

    def _call_llm(self) -> str:
        """call the LLM with the current message history"""
        response = self.llm.chat.completions.create(
            model=cfg.agent.llm_model,
            messages=self.memory.short_term.to_llm_format(),
            temperature=cfg.agent.temperature,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    def _parse_action(self, response: str):
        """
        parse the Action and Action Input from the LLM's response.
        the LLM sometimes formats things slightly differently so we
        try a few patterns before giving up.
        """
        # look for "Action: tool_name"
        action_match = re.search(r"Action:\s*(\w+)", response)
        if not action_match:
            return None, None

        action = action_match.group(1).strip()

        # look for "Action Input: {...}" (JSON) or "Action Input: plain text"
        input_match = re.search(r"Action Input:\s*(\{.*?\})", response, re.DOTALL)
        if input_match:
            try:
                # fix common Llama quirk: {key: "value"} instead of {"key": "value"}
                raw = input_match.group(1)
                raw = re.sub(r'(\w+)\s*:', r'"\1":', raw)  # quote unquoted keys
                raw = re.sub(r'""(\w+)""', r'"\1"', raw)   # fix double-quoting
                action_input = json.loads(raw)
            except json.JSONDecodeError:
                # still broken — pull out the value and guess the param name
                val_match = re.search(r'["\'](.+?)["\']', input_match.group(1))
                if val_match:
                    # use the tool's first parameter name as the key
                    tool = self.tools.get(action)
                    first_param = list(tool.input_schema.keys())[0] if tool else "query"
                    action_input = {first_param: val_match.group(1)}
                else:
                    action_input = {"query": input_match.group(1)}
        else:
            # no JSON found — grab whatever text comes after "Action Input:"
            plain_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", response)
            if plain_match:
                tool = self.tools.get(action)
                first_param = list(tool.input_schema.keys())[0] if tool else "query"
                action_input = {first_param: plain_match.group(1).strip().strip('"')}
            else:
                action_input = {}

        return action, action_input
