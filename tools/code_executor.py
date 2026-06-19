"""
tools/code_executor.py — lets the agent write and run Python code

this is honestly the most powerful tool in the kit. the agent can do math,
data processing, string manipulation, whatever — just by writing code.

security note: running arbitrary code is obviously dangerous in production.
here we use a restricted globals dict to block the most dangerous stuff.
for a real deployment you'd want Docker or a proper sandbox.
"""

import sys
import io
import signal
from contextlib import contextmanager
from config import cfg


@contextmanager
def _timeout(seconds: int):
    """
    context manager that raises TimeoutError if code takes too long.
    only works on Unix — on Windows just skip the timeout.
    """
    def _handler(signum, frame):
        raise TimeoutError(f"Code timed out after {seconds}s")

    if sys.platform != "win32":
        old = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    else:
        yield  # no timeout on Windows, live dangerously


def execute_python(code: str) -> str:
    """
    execute a Python code snippet and return stdout + any errors.

    the agent uses this for:
    - math calculations that are too complex for the calculator tool
    - string processing / formatting
    - data manipulation
    - anything it needs to "think with code"
    """
    # capture stdout so we can return it as a string
    stdout_capture = io.StringIO()

    # restricted globals — block stuff that could cause real damage
    # __builtins__ is a dict here so we can selectively allow things
    safe_globals = {
        "__builtins__": {
            "print": print,
            "range": range,
            "len": len,
            "int": int,
            "float": float,
            "str": str,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "bool": bool,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "sorted": sorted,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "isinstance": isinstance,
            "type": type,
            "repr": repr,
            "pow": pow,
            "divmod": divmod,
        },
        # give it access to math — agents use this a lot
        "__import__": __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__,
    }

    local_vars = {}

    try:
        with _timeout(cfg.tools.code_timeout):
            # redirect stdout to our capture buffer
            old_stdout = sys.stdout
            sys.stdout = stdout_capture
            try:
                exec(code, safe_globals, local_vars)
            finally:
                sys.stdout = old_stdout

        output = stdout_capture.getvalue()

        # if no print statements, check if last line is an expression with a value
        if not output.strip() and local_vars:
            # return the last assigned variable as a convenience
            last_val = list(local_vars.values())[-1]
            output = repr(last_val)

        if not output.strip():
            return "Code ran successfully (no output)."

        # limit lines so the agent doesn't get overwhelmed
        lines = output.strip().split("\n")
        if len(lines) > cfg.tools.code_output_limit:
            lines = lines[:cfg.tools.code_output_limit]
            lines.append(f"... (truncated, {len(lines)} lines shown)")

        return "\n".join(lines)

    except TimeoutError as e:
        return f"Execution timed out: {e}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
