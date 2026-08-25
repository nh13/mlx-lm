# Copyright © 2025 Apple Inc.

"""Grammar-constrained decoding for DeepSeek-V4 DSML tool calls.

Quantization damages a model's ability to emit the DSML *structural keywords*:
a 4/5-bit DeepSeek-V4-Flash generates correct tool content (names, values) and
the ``｜DSML｜`` special token itself, but — in a real multi-tool coding context,
even at temperature 0 (greedy) — corrupts the keyword tokens around it
(``tool_calls`` -> ``tool_ctools``, ``invoke`` -> ``invissue``, dropped
``｜DSML｜`` prefixes). The corrupted markers no longer match the parser, so
``tool_calls`` comes back empty and the agent loop dies. The single-tool
``get_weather`` probe never stressed this; a ``read``/``bash``/``edit`` coding
loop does.

This module fixes that at decode time. It installs an mlx-lm *logits processor*
(``processor(tokens, logits) -> logits``) that masks the model's logits so the
generated stream is *forced* to match the exact grammar ``deepseek_v4`` parses::

    <｜DSML｜tool_calls>
    <｜DSML｜invoke name="NAME">
    <｜DSML｜parameter name="PARAM" string="true|false">VALUE</｜DSML｜parameter>
    ...
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>

Only the *free* regions are left to the model: the tool name (constrained to the
registered tools), the parameter names (constrained to the tool's schema), and
the parameter values. Every structural keyword, quote, and ``string=`` type flag
is forced, so corruption is impossible and the output is guaranteed to parse.
Parameter *types* are forced from the schema (``string`` -> ``string="true"``,
otherwise ``string="false"``).

``tool_choice`` semantics (OpenAI-compatible):

* ``"required"`` / ``{"type": "function", "function": {"name": N}}`` -- *force* a
  tool call (optionally a specific tool ``N``). This is the mode that rescues a
  degraded model. When the prompt opens a thinking block (the ``deepseek_v4``
  template ends a thinking-mode prompt with ``<think>``), the block is forced
  only *after* the model closes ``</think>`` -- forcing from token 0 would
  truncate the model's reasoning and yield an empty ``finish=stop``. In chat mode
  the prompt already ends with ``</think>``, so the block is forced immediately.
* ``"auto"`` -- let the model answer in prose until it emits the ``｜DSML｜``
  special token (which it reliably gets right, and which only appears when it
  intends a tool call); from that point the block is forced.
* ``"none"`` -- this processor is never constructed.

Value handling: a value ends when the model starts the close tag (its first
``<`` in the value region) or after a runaway cap; ``｜DSML｜`` is masked out of
the value region so it cannot appear mid-value. This is robust for the usual
path/command/scalar arguments; a value that must itself contain ``<`` (e.g. a
code blob for a ``write`` tool) is the one shape this simple terminator does not
handle and would need a schema-aware (xgrammar) matcher at the marked hook.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

import mlx.core as mx

# The core DSML delimiter special token; the model emits this reliably.
_DSML = "｜DSML｜"

# Structural literals (encoded to token ids against the live tokenizer). Newlines
# match the deepseek_v4 chat-template / reference encoder layout.
_OPEN = f"<{_DSML}tool_calls>\n"
_INV_HEAD = f'<{_DSML}invoke name="'
_INV_NAME_CLOSE = '">\n'  # after the tool name
_PARAM_HEAD = f'<{_DSML}parameter name="'
_PARAM_STR_TRUE = '" string="true">'
_PARAM_STR_FALSE = '" string="false">'
_PARAM_CLOSE = f"</{_DSML}parameter>\n"
_INV_CLOSE = f"</{_DSML}invoke>\n"
_CALLS_CLOSE = f"</{_DSML}tool_calls>"

_MASK_FILL = -1e9  # finite so logsumexp stays defined even with one token allowed
_VALUE_MAX_TOKENS = 512  # runaway guard for a free value region


def _encode(tokenizer, text: str) -> Tuple[int, ...]:
    """Token ids for ``text`` with no added BOS/EOS -- matches server delimiters."""
    return tuple(tokenizer.encode(text, add_special_tokens=False))


def _last_id(tokenizer, text: str) -> Optional[int]:
    """The final token id of ``text`` (its whole id if a single token), or ``None``.

    Used to detect a marker's completion in the generated stream: a special token
    like ``</think>`` encodes to a single id, and even if it were multi-token, the
    marker is complete exactly when its last token appears.
    """
    ids = _encode(tokenizer, text)
    return ids[-1] if ids else None


def _suffix_after(seg: Tuple[int, ...], prefix: Tuple[int, ...]) -> Tuple[int, ...]:
    """The tail of ``seg`` after the shared ``prefix`` (both from the same encode).

    Used to force the remainder of a literal once the model has already emitted a
    shared opening (``<`` or ``</`` or ``<｜DSML｜``) on its own.
    """
    return seg[len(prefix):]


def _tool_name(tool: Any) -> str:
    if isinstance(tool, dict):
        fn = tool.get("function", tool)
        return fn.get("name") if isinstance(fn, dict) else str(fn)
    return str(tool)


def _tool_params(tool: Any) -> List[Tuple[str, bool]]:
    """Return ``[(param_name, is_string), ...]`` from an OpenAI tool schema."""
    fn = tool.get("function", tool) if isinstance(tool, dict) else {}
    params = (fn.get("parameters") or {}) if isinstance(fn, dict) else {}
    props = params.get("properties") or {}
    out: List[Tuple[str, bool]] = []
    for name, spec in props.items():
        typ = spec.get("type") if isinstance(spec, dict) else None
        out.append((name, typ == "string"))
    return out


def _named_choice(tool_choice: Any) -> Optional[str]:
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            return fn["name"]
    return None


class _MaskCache:
    """Caches additive logit masks as on-device ``mx`` arrays, keyed by id-set.

    A logits processor must return a full-vocab-shaped array every step, so one
    vocab-sized op per token is unavoidable. The cost this removes is (a) building
    that vector in ``numpy`` and copying it host->device *every* token (the old
    ``np.full((vocab,), ...)`` path forced a host allocation and a device sync per
    step), and (b) rebuilding the handful of *constant* masks that recur for the
    bulk of a call -- the value-region ``｜DSML｜`` deny mask (one per value token,
    and values dominate the token count) and the ``<`` / ``</`` open-or-close
    choice. Each distinct mask is materialized once, on-device, in the logits
    dtype, then reused; the number of distinct masks in a tool-call grammar is
    small (structural tokens + a few name/param positions), so the cache stays
    tiny and never needs eviction.
    """

    def __init__(self, vocab: int, dtype: Any) -> None:
        self._vocab = vocab
        self._dtype = dtype
        self._allow: dict = {}
        self._deny: dict = {}

    def _clamp(self, ids) -> frozenset:
        return frozenset(i for i in ids if 0 <= i < self._vocab)

    def allow(self, ids) -> mx.array:
        """Additive mask: ``0`` at ``ids``, ``_MASK_FILL`` everywhere else."""
        key = self._clamp(ids)
        mask = self._allow.get(key)
        if mask is None:
            mask = mx.full((self._vocab,), _MASK_FILL, dtype=self._dtype)
            if key:
                mask[mx.array(sorted(key))] = 0.0
            mx.eval(mask)  # materialize once so reuse is a leaf, not a rebuilt graph
            self._allow[key] = mask
        return mask

    def deny(self, ids) -> mx.array:
        """Additive mask: ``_MASK_FILL`` at ``ids``, ``0`` everywhere else."""
        key = self._clamp(ids)
        mask = self._deny.get(key)
        if mask is None:
            mask = mx.zeros((self._vocab,), dtype=self._dtype)
            if key:
                mask[mx.array(sorted(key))] = _MASK_FILL
            mx.eval(mask)
            self._deny[key] = mask
        return mask


class _ToolGrammarState:
    """Per-request decode state backing one logits processor.

    A small phase machine. ``force`` phases emit a fixed token segment; the
    handful of free phases (``decide``, ``name``, ``pname``, ``value``) let the
    model choose within a constrained set. Structural decisions (another
    parameter vs. close the invoke; another invoke vs. close the block) are made
    by the model but constrained to the two valid continuations.
    """

    def __init__(self, grammar: "ToolCallGrammar") -> None:
        self._g = grammar
        self._prompt_len: Optional[int] = None
        self._consumed = 0
        self._seg: Tuple[int, ...] = ()
        self._then: Optional[str] = None
        # free-region working state
        self._name_cands: List[Tuple[int, ...]] = []
        self._name_pos = 0
        self._cur_params: List[Tuple[str, bool]] = []
        self._pname_cands: List[Tuple[int, ...]] = []
        self._pname_pos = 0
        self._pname_map: dict = {}
        self._pending_is_str = False
        self._value_tokens = 0
        self._cache: Optional[_MaskCache] = None  # lazily built once vocab+dtype known
        # The concrete start phase for ``force`` mode is resolved on the first
        # call (``_on_prompt``), once we can see whether the prompt sits inside an
        # open ``<think>`` block. ``decide`` is the auto-mode default until then.
        self._phase = "decide"

    # -- masking helpers --------------------------------------------------
    def _mask_cache(self, logits: mx.array) -> _MaskCache:
        if self._cache is None:
            self._cache = _MaskCache(logits.shape[-1], logits.dtype)
        return self._cache

    def _allow(self, logits: mx.array, allowed) -> mx.array:
        return logits + self._mask_cache(logits).allow(allowed)

    def _deny(self, logits: mx.array, denied) -> mx.array:
        return logits + self._mask_cache(logits).deny(denied)

    # -- phase entry helpers ---------------------------------------------
    def _begin_force(self, seg: Tuple[int, ...], then: str) -> None:
        self._phase = "force"
        self._seg = seg
        self._then = then

    def _begin_name(self) -> None:
        self._phase = "name"
        self._name_pos = 0
        self._name_cands = list(self._g.name_token_seqs)

    def _begin_pname(self) -> None:
        # constrain parameter name to the current tool's remaining schema params
        self._phase = "pname"
        self._pname_pos = 0
        self._pname_map = {n: is_s for (n, is_s) in self._cur_params}
        self._pname_cands = [self._g.encode_seq(n) for (n, _) in self._cur_params]

    def _begin_value(self, is_str: bool) -> None:
        self._phase = "value"
        self._value_tokens = 0

    # -- main entry -------------------------------------------------------
    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        n = int(tokens.shape[-1])
        if self._prompt_len is None:
            self._prompt_len = n
            self._on_prompt(tokens)
        generated = n - self._prompt_len
        while self._consumed < generated:
            self._advance(int(tokens[self._prompt_len + self._consumed]))
            self._consumed += 1
        return self._mask(logits)

    def _on_prompt(self, tokens: mx.array) -> None:
        """Resolve the ``force``-mode start phase from the prompt's tail.

        A thinking-mode prompt ends with ``<think>`` (generation begins inside the
        reasoning block); a chat-mode prompt ends with ``</think>``. In the former
        we wait for the model to close ``</think>`` before forcing the block; in
        the latter (and whenever the marker can't be resolved) we force at once.
        Auto mode is unaffected -- it stays in ``decide`` until the model emits
        ``｜DSML｜`` on its own.
        """
        if not self._g.force:
            return
        last = int(tokens[-1]) if int(tokens.shape[-1]) else -1
        in_think = self._g.think_start_id is not None and last == self._g.think_start_id
        if in_think and self._g.think_end_id is not None:
            self._phase = "await_think"
        else:
            self._begin_force(self._g.open, then="inv_head")

    # -- transitions ------------------------------------------------------
    def _advance(self, tok: int) -> None:
        p = self._phase
        if p in ("done", "prose"):
            return
        if p == "await_think":
            if tok == self._g.think_end_id:
                # model closed </think>; now force the tool-call block
                self._begin_force(self._g.open, then="inv_head")
            return
        if p == "decide":
            if tok == self._g.dsml_id:
                # model just emitted `<｜DSML｜`; force the rest of the open block
                self._begin_force(self._g.open_suffix, then="inv_head")
            return
        if p == "force":
            self._seg = self._seg[1:]
            if not self._seg:
                self._enter(self._then)
            return
        if p == "name":
            self._name_cands = [
                s for s in self._name_cands
                if len(s) > self._name_pos and s[self._name_pos] == tok
            ]
            self._name_pos += 1
            if not any(len(s) > self._name_pos for s in self._name_cands):
                self._resolve_tool()
                self._begin_force(self._g.inv_name_close, then="pci")
            return
        if p == "pci":
            if tok == self._g.lt_id:  # `<` -> another parameter (open tag)
                self._begin_force(self._g.param_head_after_lt, then="pname")
            else:  # `</` -> close the invoke (close tag)
                self._begin_force(self._g.inv_close_after_close, then="icc")
            return
        if p == "icc":
            if tok == self._g.lt_id:  # `<` -> another invoke (open tag)
                self._begin_force(self._g.inv_head_after_lt, then="name")
            else:  # `</` -> close the tool_calls block (close tag)
                self._begin_force(self._g.calls_close_after_close, then="done")
            return
        if p == "pname":
            self._pname_cands = [
                s for s in self._pname_cands
                if len(s) > self._pname_pos and s[self._pname_pos] == tok
            ]
            self._pname_pos += 1
            if not any(len(s) > self._pname_pos for s in self._pname_cands):
                self._resolve_pname()
                seg = (self._g.param_str_true if self._pending_is_str
                       else self._g.param_str_false)
                self._begin_force(seg, then="value")
            return
        if p == "value":
            self._value_tokens += 1
            if tok == self._g.close_id:  # `</` begins the parameter close tag
                self._begin_force(self._g.param_close_after_close, then="pci")
            elif self._value_tokens >= _VALUE_MAX_TOKENS:
                self._begin_force(self._g.param_close_full, then="pci")
            return

    def _enter(self, phase: Optional[str]) -> None:
        if phase == "inv_head":
            self._begin_force(self._g.inv_head, then="name")
        elif phase == "name":
            self._begin_name()
        elif phase == "pci":
            self._phase = "pci"
        elif phase == "icc":
            self._phase = "icc"
        elif phase == "pname":
            self._begin_pname()
        elif phase == "value":
            self._begin_value(self._pending_is_str)
        else:
            self._phase = "done"

    def _resolve_tool(self) -> None:
        matched = [s for s in self._name_cands if len(s) == self._name_pos]
        seq = matched[0] if matched else (self._name_cands[0] if self._name_cands else ())
        name = self._g.decode(list(seq)).strip()
        self._cur_params = self._g.params_by_tool.get(name, [])

    def _resolve_pname(self) -> None:
        matched = [s for s in self._pname_cands if len(s) == self._pname_pos]
        seq = matched[0] if matched else (self._pname_cands[0] if self._pname_cands else ())
        pname = self._g.decode(list(seq)).strip()
        self._pending_is_str = self._pname_map.get(pname, True)
        # a parameter is emitted at most once
        self._cur_params = [(n, s) for (n, s) in self._cur_params if n != pname]

    # -- per-step mask ----------------------------------------------------
    def _mask(self, logits: mx.array) -> mx.array:
        p = self._phase
        if p in ("decide", "await_think", "done", "prose"):
            return logits
        if p == "force":
            return self._allow(logits, (self._seg[0],)) if self._seg else logits
        if p == "name":
            allowed = {s[self._name_pos] for s in self._name_cands
                       if len(s) > self._name_pos}
            return self._allow(logits, allowed) if allowed else logits
        if p == "pname":
            allowed = {s[self._pname_pos] for s in self._pname_cands
                       if len(s) > self._pname_pos}
            return self._allow(logits, allowed) if allowed else logits
        if p in ("pci", "icc"):
            # model chooses: `<` opens the next element, `</` closes this one
            return self._allow(logits, (self._g.lt_id, self._g.close_id))
        if p == "value":
            # free, but never let ｜DSML｜ appear inside a value
            return self._deny(logits, (self._g.dsml_id,))
        return logits


class ToolCallGrammar:
    """Builds a DeepSeek-V4 DSML tool-call constrained-decoding logits processor."""

    def __init__(self, tokenizer, tools: List[Any], tool_choice: Any = "auto"):
        self._tok = tokenizer
        self.open = _encode(tokenizer, _OPEN)
        self.inv_head = _encode(tokenizer, _INV_HEAD)
        self.inv_name_close = _encode(tokenizer, _INV_NAME_CLOSE)
        self.param_head = _encode(tokenizer, _PARAM_HEAD)
        self.param_str_true = _encode(tokenizer, _PARAM_STR_TRUE)
        self.param_str_false = _encode(tokenizer, _PARAM_STR_FALSE)
        self.param_close = _encode(tokenizer, _PARAM_CLOSE)
        self.inv_close = _encode(tokenizer, _INV_CLOSE)
        self.calls_close = _encode(tokenizer, _CALLS_CLOSE)

        # Single-token anchors the model emits reliably. Open tags begin with
        # `<` (id for "<"); close tags begin with `</`, which is ONE token in
        # this vocab (distinct from "<" and "/") -- the crux of the boundary.
        self.dsml_id = _encode(tokenizer, _DSML)[0]  # ｜DSML｜
        self.lt_id = _encode(tokenizer, "<")[0]      # `<`  opens a tag
        self.close_id = _encode(tokenizer, "</")[0]  # `</` opens a close tag

        # Thinking-block markers (single special tokens in the DeepSeek vocab);
        # used only in force mode to defer forcing until the model closes </think>.
        self.think_start_id = _last_id(tokenizer, "<think>")
        self.think_end_id = _last_id(tokenizer, "</think>")

        # Suffixes forced after the model emits the opener token on its own.
        self.param_head_after_lt = _suffix_after(self.param_head, (self.lt_id,))
        self.inv_head_after_lt = _suffix_after(self.inv_head, (self.lt_id,))
        self.inv_close_after_close = _suffix_after(self.inv_close, (self.close_id,))
        self.calls_close_after_close = _suffix_after(self.calls_close, (self.close_id,))
        self.param_close_after_close = _suffix_after(self.param_close, (self.close_id,))
        self.param_close_full = self.param_close
        self.open_suffix = _suffix_after(self.open, (self.lt_id, self.dsml_id))

        forced_name = _named_choice(tool_choice)
        names = [_tool_name(t) for t in (tools or [])]
        if forced_name is not None:
            names = [n for n in names if n == forced_name] or names[:1]
        self.name_token_seqs = [self.encode_seq(n) for n in names]
        self.params_by_tool = {_tool_name(t): _tool_params(t) for t in (tools or [])}

        self.force = tool_choice == "required" or forced_name is not None

    def encode_seq(self, text: str) -> Tuple[int, ...]:
        return _encode(self._tok, text)

    def decode(self, ids: List[int]) -> str:
        return self._tok.decode(ids)

    def logits_processor(self) -> Callable[[mx.array, mx.array], mx.array]:
        return _ToolGrammarState(self)
