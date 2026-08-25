# Copyright © 2025 Apple Inc.

"""Grammar-constrained decoding for DeepSeek-family tool calls.

Quantization damages a model's ability to emit rare *special* tokens: a 4-bit
DeepSeek-V4-Flash generates correct tool *content* (names, JSON arguments) but
cannot reliably emit the tool-call delimiters — ``<｜tool▁calls▁begin｜>`` and
friends — instead producing corrupted / wrong-format text, so the server's
token-id state machine never enters the ``tool`` region and ``tool_calls`` comes
back empty. (Higher quants remove the *corruption* but not the *format choice*:
DeepSeek-V4-Flash still defaults to a generic markdown-JSON block rather than its
native special-token format at every tested quant.)

This module fixes that at decode time. It installs an mlx-lm *logits processor*
(``processor(tokens, logits) -> logits``) that masks the model's logits so the
generated stream is *forced* to match the exact grammar ``deepseek_v3`` parses::

    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME
    ```json
    {ARGS}
    ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>

The structural tokens (which the model cannot emit unaided) are *forced*; the
name is constrained to the set of registered tools; the argument region is left
to the model (it produces valid JSON reliably) but its completion is detected so
the closing fence and delimiters are forced. The result is guaranteed to parse.

``tool_choice`` semantics (OpenAI-compatible):

* ``"required"`` / ``{"type": "function", "function": {"name": N}}`` — *force* a
  tool call (optionally a specific tool ``N``). This is the mode that rescues a
  degraded model: it can no longer fail to start the call. For DeepSeek-V4-Flash
  this is the only effective mode, since the model never starts a native call on
  its own.
* ``"auto"`` — *offer* the tool branch. Because ``<｜tool▁calls▁begin｜>`` is a
  single token, the choice is exactly one decision at the first generated token:
  if the model emits that token we constrain the rest of the block; otherwise we
  step aside and let it answer in prose. (A model that never emits the token —
  like DeepSeek-V4-Flash — simply answers in prose, unconstrained.)
* ``"none"`` — this processor is never constructed.

The JSON-argument region enforces *well-formedness* (balanced, terminated),
which suffices because the model emits schema-valid JSON on its own. For strict
schema enforcement (types, enums, required keys) an ``xgrammar`` matcher can be
substituted at the marked hook in ``_ToolGrammarState`` (see the module tests
and project notes for the integration point).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import mlx.core as mx
import numpy as np

# Structural literals of the DeepSeek tool-call grammar, encoded to token ids
# against the live tokenizer at construction. Each special token is a single id
# in the DeepSeek vocab, but arbitrary-length encodings are handled uniformly.
_CALLS_BEGIN = "<｜tool▁calls▁begin｜>"
_CALLS_END = "<｜tool▁calls▁end｜>"
_CALL_HEAD = "<｜tool▁call▁begin｜>function<｜tool▁sep｜>"  # up to (not incl.) NAME
_FENCE_OPEN = "\n```json\n"  # follows NAME
_FENCE_CLOSE = "\n```<｜tool▁call▁end｜>"  # follows ARGS
_MASK_FILL = -1e9  # finite so logsumexp stays defined even with one token allowed
_JSON_MAX_TOKENS = 512  # runaway guard for the free JSON region


def _encode(tokenizer, text: str) -> tuple[int, ...]:
    """Token ids for ``text`` with no added BOS/EOS — matches server delimiters."""
    return tuple(tokenizer.encode(text, add_special_tokens=False))


class _JsonCompletion:
    """Character-level tracker reporting when a top-level JSON value is complete.

    Only *sampled* tokens are fed here (one ``decode`` per step), so it is cheap.
    It detects a balanced, terminated JSON value so the closing fence can be
    forced the instant arguments finish; it does not validate against a schema.
    """

    def __init__(self) -> None:
        self._depth = 0
        self._in_string = False
        self._escape = False
        self._started = False  # seen the first non-whitespace char yet?
        self._scalar = False  # top-level value is a bare scalar (string/number/…)
        self.done = False

    def feed(self, text: str) -> None:
        for ch in text:
            if self.done:
                return
            if self._in_string:
                if self._escape:
                    self._escape = False
                elif ch == "\\":
                    self._escape = True
                elif ch == '"':
                    self._in_string = False
                    if self._scalar and self._depth == 0:
                        self.done = True
                continue
            if ch in " \t\r\n":
                if self._scalar and self._depth == 0 and self._started:
                    self.done = True  # whitespace terminates a bare scalar
                continue
            if not self._started:
                self._started = True
                self._scalar = ch not in "{["
            if ch == '"':
                self._in_string = True
            elif ch in "{[":
                self._depth += 1
            elif ch in "}]":
                self._depth -= 1
                if self._depth == 0:
                    self.done = True
            elif self._scalar and self._depth == 0 and ch == ",":
                self.done = True


class _ToolGrammarState:
    """Per-request decode state. One instance backs one logits processor.

    Phases:
      ``decide``       (auto only) let the model pick the first token
      ``force_begin``  force ``<｜tool▁calls▁begin｜>``
      ``head``         force ``<｜tool▁call▁begin｜>function<｜tool▁sep｜>``
      ``name``         constrain to a registered tool name
      ``fence_open``   force ``\\n```json\\n``
      ``json``         free content; completion detected from sampled tokens
      ``fence_close``  force ``\\n```<｜tool▁call▁end｜>``
      ``calls_end``    force ``<｜tool▁calls▁end｜>``
      ``done``/``prose`` no further constraint
    """

    def __init__(self, grammar: "ToolCallGrammar") -> None:
        self._g = grammar
        self._prompt_len: Optional[int] = None
        self._consumed = 0  # generated tokens already advanced through the grammar
        if grammar.force:
            self._phase = "force_begin"
            self._seg: tuple[int, ...] = grammar.calls_begin
        else:
            self._phase = "decide"
            self._seg = ()
        self._name_candidates = list(grammar.name_token_seqs)
        self._name_pos = 0
        self._json: Optional[_JsonCompletion] = None
        self._json_tokens = 0

    # -- masking helpers --------------------------------------------------
    def _allow(self, logits: mx.array, allowed) -> mx.array:
        """Return logits with everything outside ``allowed`` driven to -inf."""
        v = logits.shape[-1]
        add = np.full((v,), _MASK_FILL, dtype=np.float32)
        idx = [t for t in allowed if 0 <= t < v]
        if idx:
            add[idx] = 0.0
        return logits + mx.array(add)[None].astype(logits.dtype)

    def _force(self, logits: mx.array) -> mx.array:
        return self._allow(logits, (self._seg[0],)) if self._seg else logits

    # -- main entry -------------------------------------------------------
    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        n = int(tokens.shape[-1])
        if self._prompt_len is None:
            self._prompt_len = n  # first call: logits are for generated token #0
        generated = n - self._prompt_len
        while self._consumed < generated:
            self._advance(int(tokens[self._prompt_len + self._consumed]))
            self._consumed += 1
        return self._mask(logits)

    # -- grammar transitions ---------------------------------------------
    def _advance(self, tok: int) -> None:
        p = self._phase
        if p in ("done", "prose"):
            return
        if p == "decide":
            if self._g.calls_begin and tok == self._g.calls_begin[0]:
                rest = self._g.calls_begin[1:]
                if rest:
                    self._phase, self._seg = "force_begin", rest
                else:
                    self._enter_head()
            else:
                self._phase = "prose"
            return
        if p in ("force_begin", "head", "fence_open", "fence_close", "calls_end"):
            self._seg = self._seg[1:]
            if not self._seg:
                self._on_forced_done(p)
            return
        if p == "name":
            self._name_candidates = [
                s for s in self._name_candidates
                if len(s) > self._name_pos and s[self._name_pos] == tok
            ]
            self._name_pos += 1
            if not any(len(s) > self._name_pos for s in self._name_candidates):
                self._phase, self._seg = "fence_open", self._g.fence_open
            return
        if p == "json":
            self._json_tokens += 1
            self._json.feed(self._g.decode([tok]))  # xgrammar hook: advance matcher
            if self._json.done or self._json_tokens >= _JSON_MAX_TOKENS:
                self._phase, self._seg = "fence_close", self._g.fence_close
            return

    def _enter_head(self) -> None:
        self._phase, self._seg = "head", self._g.call_head

    def _on_forced_done(self, phase: str) -> None:
        if phase == "force_begin":
            self._enter_head()
        elif phase == "head":
            self._phase = "name"
            self._name_pos = 0
            self._name_candidates = list(self._g.name_token_seqs)
        elif phase == "fence_open":
            self._phase = "json"
            self._json = _JsonCompletion()
            self._json_tokens = 0
        elif phase == "fence_close":
            self._phase, self._seg = "calls_end", self._g.calls_end
        elif phase == "calls_end":
            self._phase = "done"

    # -- per-step mask ----------------------------------------------------
    def _mask(self, logits: mx.array) -> mx.array:
        p = self._phase
        if p in ("decide", "json", "prose", "done"):
            return logits  # unconstrained (xgrammar hook: mask json region here)
        if p in ("force_begin", "head", "fence_open", "fence_close", "calls_end"):
            return self._force(logits)
        if p == "name":
            allowed = {
                s[self._name_pos]
                for s in self._name_candidates
                if len(s) > self._name_pos
            }
            return self._allow(logits, allowed) if allowed else logits
        return logits


class ToolCallGrammar:
    """Builds a DeepSeek tool-call constrained-decoding logits processor.

    Args:
        tokenizer: a loaded mlx-lm ``TokenizerWrapper`` (or HF tokenizer), used
            for ``encode`` / ``decode``.
        tools: the request's OpenAI-style ``tools`` list.
        tool_choice: ``"auto"``, ``"required"``, ``"none"``, or a
            ``{"type": "function", "function": {"name": ...}}`` dict.
    """

    def __init__(self, tokenizer, tools: list[Any], tool_choice: Any = "auto"):
        self._tok = tokenizer
        self.calls_begin = _encode(tokenizer, _CALLS_BEGIN)
        self.calls_end = _encode(tokenizer, _CALLS_END)
        self.call_head = _encode(tokenizer, _CALL_HEAD)
        self.fence_open = _encode(tokenizer, _FENCE_OPEN)
        self.fence_close = _encode(tokenizer, _FENCE_CLOSE)

        forced_name = _named_choice(tool_choice)
        names = [_tool_name(t) for t in (tools or [])]
        if forced_name is not None:
            names = [n for n in names if n == forced_name] or names[:1]
        self.name_token_seqs = [_encode(tokenizer, n) for n in names]

        self.force = tool_choice == "required" or forced_name is not None

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)

    def logits_processor(self) -> Callable[[mx.array, mx.array], mx.array]:
        """Return a fresh stateful ``processor(tokens, logits) -> logits``."""
        return _ToolGrammarState(self)


# --------------------------------------------------------------------------
# small module-scope helpers (kept out of the class for direct testing)
# --------------------------------------------------------------------------
def _tool_name(tool: Any) -> str:
    if isinstance(tool, dict):
        fn = tool.get("function", tool)
        return fn.get("name") if isinstance(fn, dict) else str(fn)
    return str(tool)


def _named_choice(tool_choice: Any) -> Optional[str]:
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            return fn["name"]
    return None
