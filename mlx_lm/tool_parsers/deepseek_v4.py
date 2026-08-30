"""DeepSeek-V4 DSML tool-call parser.

DeepSeek-V4 models emit tool calls in DSML (DeepSeek Markup Language):

    <｜DSML｜tool_calls>
    <｜DSML｜invoke name="TOOL_NAME">
    <｜DSML｜parameter name="PARAM" string="true|false">VALUE</｜DSML｜parameter>
    ...
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>

``string="true"`` marks a raw string value; ``string="false"`` marks a JSON
value (number, boolean, array, object) — mirroring the encode side in
``chat_templates/deepseek_v4`` / ``deepseek_v32.encode_arguments_to_dsml``.

The invoke close tag is matched leniently (``</｜DSML｜inv…>``): the models
systematically truncate it to ``</｜DSML｜inv>`` instead of the canonical
``</｜DSML｜invoke>`` (reproducible across the 4/5/6-bit quants), and the strict
reference parser rejects that form. Keeping the leniency inside the invoke
regex scopes it to the tag boundary rather than rewriting the whole text.

Wiring: set ``tool_parser_type: "deepseek_v4"`` in the model's
``tokenizer_config.json`` (paired with ``chat_template_type: "deepseek_v4"``).
"""

import json
import re
from typing import Any, Dict, List

from mlx_lm.chat_templates.deepseek_v32 import dsml_token

# The core DSML delimiter (a single special token, id 128825 in the vocab);
# shared with the chat template so encode and decode cannot drift.
_DSML = dsml_token

# Markers the server's state machine keys on to capture the tool-call block.
# Anchored WITHOUT the trailing '>': at generation the '>' merges with the next
# byte into a different token, breaking exact token-sequence matching (mlx-lm #984).
tool_call_start = f"<{_DSML}tool_calls"
tool_call_end = f"</{_DSML}tool_calls"

# The invoke close is lenient (``inv[^>]*>``) to tolerate the models' truncated
# ``</｜DSML｜inv>`` tag; the non-greedy body keeps it scoped to the tag boundary.
_INVOKE_RE = re.compile(
    r"<" + re.escape(_DSML) + r'invoke\s+name="(?P<name>.*?)"\s*>'
    r"(?P<body>.*?)</" + re.escape(_DSML) + r"inv[^>]*>",
    re.DOTALL,
)
_PARAM_RE = re.compile(
    r"<" + re.escape(_DSML) + r'parameter\s+name="(?P<name>.*?)"\s+'
    r'string="(?P<is_str>true|false)"\s*>(?P<value>.*?)</'
    + re.escape(_DSML)
    + r"parameter>",
    re.DOTALL,
)


def _decode_value(is_str: str, value: str) -> Any:
    """Decode a DSML parameter value honoring its ``string`` type flag."""
    if is_str == "true":
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        # Fall back to the raw text if a string="false" value is not valid JSON.
        return value


def parse_tool_call(text: str, tools: Any = None) -> List[Dict[str, Any]]:
    """Parse a captured DSML tool-call block into mlx tool-call dicts.

    Args:
        text: The captured tool-call text (with or without the outer
            ``<｜DSML｜tool_calls>`` markers). Extraction is anchored on the
            ``invoke``/``parameter`` tags, so surrounding text is tolerated.
        tools: The registered tools (unused; present for the parser contract).

    Returns:
        A list of ``{"name": str, "arguments": dict}`` — one per invoke.

    Raises:
        ValueError: If no well-formed invoke block is found, so the caller
            (``ToolCallFormatter``) can treat the text as unparsed.
    """
    calls: List[Dict[str, Any]] = []
    for invoke in _INVOKE_RE.finditer(text):
        name = invoke.group("name").strip()
        arguments: Dict[str, Any] = {}
        for param in _PARAM_RE.finditer(invoke.group("body")):
            arguments[param.group("name").strip()] = _decode_value(
                param.group("is_str"), param.group("value")
            )
        calls.append({"name": name, "arguments": arguments})
    if not calls:
        raise ValueError("no DSML invoke block found in tool-call text")
    return calls
