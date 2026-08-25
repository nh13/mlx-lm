"""Tests for DSML grammar-constrained decoding (``mlx_lm.tool_grammar``).

The grammar forces the DSML tool-call structure so a quantization-degraded model
cannot corrupt the structural keywords. A scripted "model" drives the phase
machine with a fixed tool call; the test asserts the forced token stream decodes
to valid DSML that the ``deepseek_v4`` parser recovers exactly.

These need the DeepSeek-V4-Flash tokenizer (the token-id boundaries -- ``<`` vs
the single ``</`` token, the ``｜DSML｜`` special token -- are vocab-specific), so
they skip when the model is not in the local HF cache. The mask-cache tests use a
tiny synthetic vocab and always run.
"""

import glob
import os
import unittest

import mlx.core as mx

from mlx_lm.tool_grammar import ToolCallGrammar, _MaskCache, _MASK_FILL
from mlx_lm.tool_parsers import deepseek_v4


def _load_tokenizer():
    bases = [
        os.environ.get("HF_HOME"),
        os.path.expanduser("~/.cache/huggingface"),
        "/Volumes/scratch-00001/hf",
    ]
    for base in bases:
        if not base:
            continue
        hits = glob.glob(
            os.path.join(base, "hub/models--mlx-community--DeepSeek-V4-Flash-4bit/snapshots/*/")
        )
        if hits:
            from transformers import AutoTokenizer

            try:
                return AutoTokenizer.from_pretrained(hits[0], trust_remote_code=True)
            except Exception:
                # Snapshot cached but its tokenizer cannot be built here (slow->fast
                # conversion needs sentencepiece/tiktoken); treat as unavailable so the
                # vocab-specific tests skip, as the module docstring intends.
                return None
    return None


_TOK = _load_tokenizer()

READ_TOOL = {
    "type": "function",
    "function": {
        "name": "read",
        "description": "Read a file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
}
WRITE_TOOL = {
    "type": "function",
    "function": {
        "name": "write",
        "description": "Write a file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
}


class _ScriptedModel:
    """A deterministic single-invoke 'model' that fills the grammar's free phases.

    Forced phases have exactly one allowed token; the free phases (name, param
    name, value, and the open-or-close decisions) are filled from a fixed tool
    call so the whole stream is valid and parseable.
    """

    def __init__(self, grammar, tool_name, params):
        self._g = grammar
        self._enc = lambda s: list(grammar.encode_seq(s))
        self._tool_name = tool_name
        self._remaining = list(params)  # [(pname, pvalue), ...]
        self._name_buf = []
        self._pname_buf = []
        self._val_buf = None
        self._cur_val = None
        self._emitted_lt = False  # decide: emit `<` before `｜DSML｜`

    def next_token(self, proc):
        ph = proc._phase
        if ph == "decide":
            if not self._emitted_lt:
                self._emitted_lt = True
                return self._g.lt_id
            return self._g.dsml_id
        if ph == "force":
            return proc._seg[0]
        if ph == "name":
            if not self._name_buf:
                self._name_buf = self._enc(self._tool_name)
            return self._name_buf.pop(0)
        if ph == "pci":  # open the next parameter, or close the invoke
            return self._g.lt_id if self._remaining else self._g.close_id
        if ph == "pname":
            if not self._pname_buf:
                pname, pval = self._remaining[0]
                self._pname_buf = self._enc(pname)
                self._cur_val = pval
                self._val_buf = None
            return self._pname_buf.pop(0)
        if ph == "value":
            if self._val_buf is None:
                self._val_buf = self._enc(self._cur_val)
            if self._val_buf:
                return self._val_buf.pop(0)
            self._remaining.pop(0)  # this parameter is complete
            self._val_buf = None
            return self._g.close_id  # `</` begins the parameter close tag
        if ph == "icc":  # single invoke -> close the block
            return self._g.close_id
        raise AssertionError(f"unexpected free phase {ph!r}")


def _drive(grammar, prompt_tokens, tool_name, params, max_steps=8000):
    """Run the scripted model to completion; return decoded generated text."""
    proc = grammar.logits_processor()
    vocab = len(_TOK)
    model = _ScriptedModel(grammar, tool_name, params)
    tokens = list(prompt_tokens)
    gen = []
    for _ in range(max_steps):
        proc(mx.array(tokens), mx.zeros((1, vocab)))
        if proc._phase == "done":
            break
        tok = int(model.next_token(proc))
        tokens.append(tok)
        gen.append(tok)
    else:
        raise AssertionError("grammar never reached 'done'")
    return _TOK.decode(gen)


READ_PARAMS = [("path", "crates/x.rs"), ("offset", "1150"), ("limit", "100")]


@unittest.skipUnless(_TOK is not None, "DeepSeek-V4-Flash tokenizer not in HF cache")
class TestGrammarForcedOutput(unittest.TestCase):
    def _assert_read_call(self, text):
        calls = deepseek_v4.parse_tool_call(text, [READ_TOOL])
        self.assertEqual(calls[0]["name"], "read")
        args = calls[0]["arguments"]
        self.assertEqual(args.get("path"), "crates/x.rs")
        self.assertEqual(args.get("offset"), 1150)
        self.assertEqual(args.get("limit"), 100)

    def test_required_mode_parses(self):
        g = ToolCallGrammar(_TOK, [READ_TOOL], tool_choice="required")
        out = _drive(g, _TOK.encode("dummy prompt", add_special_tokens=False), "read", READ_PARAMS)
        self._assert_read_call(out)

    def test_auto_mode_parses(self):
        g = ToolCallGrammar(_TOK, [READ_TOOL], tool_choice="auto")
        out = _drive(g, _TOK.encode("dummy prompt", add_special_tokens=False), "read", READ_PARAMS)
        self._assert_read_call(out)

    def test_named_choice_forces_that_tool(self):
        g = ToolCallGrammar(
            _TOK, [READ_TOOL, WRITE_TOOL],
            tool_choice={"type": "function", "function": {"name": "read"}},
        )
        out = _drive(g, _TOK.encode("dummy prompt", add_special_tokens=False), "read", READ_PARAMS)
        self._assert_read_call(out)

    def test_await_think_defers_forcing_until_think_close(self):
        """In force mode, a prompt ending inside a ``<think>`` block must not
        force the tool-call block until the model closes ``</think>``."""
        g = ToolCallGrammar(_TOK, [READ_TOOL], tool_choice="required")
        self.assertIsNotNone(g.think_start_id)
        self.assertIsNotNone(g.think_end_id)
        proc = g.logits_processor()
        vocab = len(_TOK)
        tokens = _TOK.encode("reason first", add_special_tokens=False) + [g.think_start_id]
        proc(mx.array(tokens), mx.zeros((1, vocab)))
        self.assertEqual(proc._phase, "await_think")
        for t in _TOK.encode(" thinking...", add_special_tokens=False):
            tokens.append(t)
            proc(mx.array(tokens), mx.zeros((1, vocab)))
            self.assertEqual(proc._phase, "await_think")
        tokens.append(g.think_end_id)
        proc(mx.array(tokens), mx.zeros((1, vocab)))
        self.assertEqual(proc._phase, "force")

    def test_chat_mode_forces_immediately(self):
        """A prompt ending with ``</think>`` (chat mode) forces at token 0."""
        g = ToolCallGrammar(_TOK, [READ_TOOL], tool_choice="required")
        proc = g.logits_processor()
        vocab = len(_TOK)
        tokens = _TOK.encode("answer", add_special_tokens=False) + [g.think_end_id]
        proc(mx.array(tokens), mx.zeros((1, vocab)))
        self.assertEqual(proc._phase, "force")

    def test_value_region_masks_dsml_token(self):
        """``｜DSML｜`` must be forbidden inside a free value so it cannot corrupt
        the block mid-argument."""
        g = ToolCallGrammar(_TOK, [READ_TOOL], tool_choice="required")
        proc = g.logits_processor()
        vocab = len(_TOK)
        tokens = _TOK.encode("dummy", add_special_tokens=False)
        model = _ScriptedModel(g, "read", [("path", "crates/x.rs")])
        for _ in range(400):
            row = mx.array(proc(mx.array(tokens), mx.zeros((1, vocab))))[0]
            if proc._phase == "value":
                self.assertLess(float(row[g.dsml_id]), _MASK_FILL / 2,
                                "｜DSML｜ should be masked out of a value")
                return
            tokens.append(int(model.next_token(proc)))
        self.fail("never reached value phase")


class TestMaskCache(unittest.TestCase):
    """Mask-cache semantics -- no tokenizer needed."""

    def test_allow_and_deny_additive_semantics(self):
        cache = _MaskCache(vocab=16, dtype=mx.float32)
        allow = cache.allow([2, 5])
        self.assertEqual(float(allow[2]), 0.0)
        self.assertEqual(float(allow[5]), 0.0)
        self.assertEqual(float(allow[0]), _MASK_FILL)
        deny = cache.deny([7])
        self.assertEqual(float(deny[7]), _MASK_FILL)
        self.assertEqual(float(deny[0]), 0.0)

    def test_masks_are_cached_by_id_set(self):
        cache = _MaskCache(vocab=16, dtype=mx.float32)
        a1 = cache.allow([3, 1])
        a2 = cache.allow([1, 3])  # same set, different order
        self.assertIs(a1, a2)
        d1 = cache.deny([9])
        d2 = cache.deny([9])
        self.assertIs(d1, d2)

    def test_out_of_range_ids_are_clamped(self):
        cache = _MaskCache(vocab=8, dtype=mx.float32)
        m = cache.allow([3, 999, -1])  # only 3 is in range
        self.assertEqual(float(m[3]), 0.0)
        self.assertEqual(float(m[0]), _MASK_FILL)


if __name__ == "__main__":
    unittest.main()
