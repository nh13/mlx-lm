"""Unit tests for the DeepSeek-V4 DSML tool-call parser.

These run without loading any model: they feed sample DSML blocks (including the
exact truncated-close-tag output captured from the 4/5/6-bit quants) to the
parser and assert the recovered tool calls.
"""

import json
import unittest

from mlx_lm.tool_parsers import deepseek_v4


class TestDeepSeekV4ToolParser(unittest.TestCase):
    def test_markers(self):
        # Anchors intentionally omit the trailing '>': at generation the '>'
        # merges with the following newline into one token, so an exact
        # token-sequence match on the full marker fails (mlx-lm #984).
        self.assertEqual(deepseek_v4.tool_call_start, "<｜DSML｜tool_calls")
        self.assertEqual(deepseek_v4.tool_call_end, "</｜DSML｜tool_calls")

    def test_wellformed_single_call(self):
        text = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="get_weather">\n'
            '<｜DSML｜parameter name="location" string="true">Paris, France</｜DSML｜parameter>\n'
            '<｜DSML｜parameter name="unit" string="true">celsius</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        )
        calls = deepseek_v4.parse_tool_call(text, [])
        self.assertEqual(
            calls,
            [
                {
                    "name": "get_weather",
                    "arguments": {"location": "Paris, France", "unit": "celsius"},
                }
            ],
        )

    def test_string_false_json_values(self):
        # string="false" => value is JSON (number, bool, array, object).
        text = (
            '<｜DSML｜invoke name="set_config">\n'
            '<｜DSML｜parameter name="count" string="false">3</｜DSML｜parameter>\n'
            '<｜DSML｜parameter name="enabled" string="false">true</｜DSML｜parameter>\n'
            '<｜DSML｜parameter name="tags" string="false">["a", "b"]</｜DSML｜parameter>\n'
            '<｜DSML｜parameter name="label" string="true">hello</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>"
        )
        calls = deepseek_v4.parse_tool_call(text, [])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "set_config")
        self.assertEqual(
            calls[0]["arguments"],
            {"count": 3, "enabled": True, "tags": ["a", "b"], "label": "hello"},
        )
        # Arguments must be a dict that json-serializes cleanly (server json.dumps it).
        json.dumps(calls[0]["arguments"])

    def test_lenient_truncated_close_tag(self):
        # Exact assistant output captured from 4/5/6-bit: close tag truncated to inv.
        text = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="get_weather">\n'
            '<｜DSML｜parameter name="location" string="true">Paris, France</｜DSML｜parameter>\n'
            '<｜DSML｜parameter name="unit" string="true">celsius</｜DSML｜parameter>\n'
            "</｜DSML｜inv>\n"
            "</｜DSML｜tool_calls>"
        )
        calls = deepseek_v4.parse_tool_call(text, [])
        self.assertEqual(
            calls,
            [
                {
                    "name": "get_weather",
                    "arguments": {"location": "Paris, France", "unit": "celsius"},
                }
            ],
        )

    def test_multiple_invokes(self):
        text = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="get_weather">\n'
            '<｜DSML｜parameter name="location" string="true">Tokyo</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            '<｜DSML｜invoke name="search">\n'
            '<｜DSML｜parameter name="query" string="true">news</｜DSML｜parameter>\n'
            "</｜DSML｜inv>\n"
            "</｜DSML｜tool_calls>"
        )
        calls = deepseek_v4.parse_tool_call(text, [])
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            calls[0], {"name": "get_weather", "arguments": {"location": "Tokyo"}}
        )
        self.assertEqual(calls[1], {"name": "search", "arguments": {"query": "news"}})

    def test_embedded_in_assistant_text(self):
        # The parser tolerates surrounding text (e.g. a trailing newline block).
        text = (
            "some preamble\n\n<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="get_weather">\n'
            '<｜DSML｜parameter name="location" string="true">Berlin</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>\n"
        )
        calls = deepseek_v4.parse_tool_call(text, [])
        self.assertEqual(
            calls, [{"name": "get_weather", "arguments": {"location": "Berlin"}}]
        )

    def test_no_invoke_raises(self):
        with self.assertRaises(ValueError):
            deepseek_v4.parse_tool_call("no tool call here", [])


if __name__ == "__main__":
    unittest.main()
