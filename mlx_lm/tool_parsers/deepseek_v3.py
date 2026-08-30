# Copyright © 2025 Apple Inc.

"""
Tool-call parser for DeepSeek V3 / V3.1 / R1 / V4 models.

Ported from vLLM's ``deepseekv3_tool_parser.py``. DeepSeek emits tool calls as::

    <｜tool▁calls▁begin｜>
    <｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME
    ```json
    {ARGS}
    ```<｜tool▁call▁end｜>
    ...one block per call...
    <｜tool▁calls▁end｜>

The server's ``SequenceStateMachine`` uses the OUTER ``calls▁begin``/``calls▁end``
pair as the tool region delimiters (registered below as ``tool_call_start`` /
``tool_call_end``). It strips those and hands the concatenation of the inner
call blocks to ``parse_tool_call``, which therefore returns a *list* of
``{"name", "arguments"}`` dicts — one per inner ``call▁begin``/``call▁end`` block.
The ToolCallFormatter already flattens a list return, so multi-call outputs work.
"""

import json
from typing import Any

import regex as re

# Outer wrapper — registered with the server as the tool-state delimiters so no
# stray tokens leak into assistant content.
tool_call_start = "<｜tool▁calls▁begin｜>"
tool_call_end = "<｜tool▁calls▁end｜>"

# Inner per-call block, parsed here rather than by the state machine. Non-greedy
# so multiple calls in one region are matched individually (vLLM uses a greedy
# variant that only handles a single call cleanly).
_TOOL_CALL_RE = re.compile(
    r"<｜tool▁call▁begin｜>(?P<type>[^<]*?)<｜tool▁sep｜>"
    r"(?P<name>.*?)\n```json\n(?P<args>.*?)\n```<｜tool▁call▁end｜>",
    re.DOTALL,
)


def _coerce_arguments(raw: str) -> dict[str, Any]:
    """Decode the JSON arguments block, tolerating minor malformation."""
    raw = raw.strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Keep the call rather than dropping it; surface the raw payload.
        return {"_raw": raw}
    return parsed if isinstance(parsed, dict) else {"_value": parsed}


def parse_tool_call(text: str, tools: list[Any] | None = None):
    """Parse a DeepSeek tool-call region into a list of ``{name, arguments}`` dicts."""
    calls = [
        dict(
            name=match.group("name").strip(),
            arguments=_coerce_arguments(match.group("args")),
        )
        for match in _TOOL_CALL_RE.finditer(text)
    ]
    if not calls:
        # Inner delimiters absent — generation was likely truncated mid-call.
        stripped = text.strip()
        if stripped:
            return [dict(name="unknown", arguments={"_raw": stripped})]
    return calls
