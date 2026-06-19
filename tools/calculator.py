"""
tools/calculator.py — safe math evaluator

yeah we could just use the code executor for this, but having a dedicated
calculator tool means the agent reaches for it on simple math instead of
writing a whole Python script. faster and cleaner.
"""

import math
import re


def calculate(expression: str) -> str:
    """
    evaluate a math expression safely.

    supports: +, -, *, /, **, %, sqrt, log, sin, cos, tan, pi, e
    examples:
        "2 ** 10"           → 1024
        "sqrt(144)"         → 12.0
        "log(100, 10)"      → 2.0
        "(15 * 8) / 3 + 7"  → 47.0
    """
    # whitelist only the characters we actually need
    # this prevents stuff like __import__ sneaking through
    allowed = re.compile(r'^[\d\s\+\-\*\/\(\)\.\,\%\^a-z\_]+$', re.IGNORECASE)
    if not allowed.match(expression.strip()):
        return "Error: expression contains invalid characters."

    # safe math namespace — only expose what's needed
    safe_env = {
        "sqrt": math.sqrt,
        "log": math.log,
        "log2": math.log2,
        "log10": math.log10,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
        "round": round,
        "pow": pow,
        "factorial": math.factorial,
        "floor": math.floor,
        "ceil": math.ceil,
    }

    try:
        result = eval(expression, {"__builtins__": {}}, safe_env)
        # round floats to 6 decimal places — avoids 0.30000000000000004 type nonsense
        if isinstance(result, float):
            result = round(result, 6)
        return str(result)
    except ZeroDivisionError:
        return "Error: division by zero."
    except Exception as e:
        return f"Error evaluating expression: {e}"
