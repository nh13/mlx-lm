# Copyright © 2025 Apple Inc.

"""Unit tests for grammar-constrained DeepSeek tool-call decoding.

Runs WITHOUT the big model: a tiny fake tokenizer + synthetic logits drive the
logits processor deterministically, and outputs are checked against the real
``mlx_lm.tool_parsers.deepseek_v3.parse_tool_call``.

Run:  ~/work/v4exp/bin/python mlx_lm/tests/test_tool_grammar.py
"""

import importlib.util
import os
import string
import sys

import mlx.core as mx

# Load the modules-under-test directly from this worktree, so we exercise the
# branch's code rather than whatever ``mlx_lm`` the editable install resolves to.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_tg = _load("_wt_tool_grammar", "mlx_lm/tool_grammar.py")
ToolCallGrammar, _JsonCompletion = _tg.ToolCallGrammar, _tg._JsonCompletion
parse_tool_call = _load("_wt_deepseek_v3", "mlx_lm/tool_parsers/deepseek_v3.py").parse_tool_call

# The five DeepSeek tool-call specials + "function" are atomic tokens; everything
# else tokenizes to single characters. That is enough to exercise the grammar.
_SPECIALS = [
    "<｜tool▁calls▁begin｜>",
    "<｜tool▁call▁begin｜>",
    "<｜tool▁sep｜>",
    "<｜tool▁call▁end｜>",
    "<｜tool▁calls▁end｜>",
    "function",
]
_JUNK = "✗"  # stands in for the ZWJ-corrupted garbage a 4-bit model emits


class FakeTokenizer:
    """Greedy longest-match tokenizer over a fixed set of atomic words + chars."""

    def __init__(self, extra_words):
        chars = list(dict.fromkeys(string.printable + _JUNK))
        self._words = sorted(set(_SPECIALS) | set(extra_words), key=len, reverse=True)
        vocab = self._words + chars
        self._to_id = {t: i for i, t in enumerate(vocab)}
        self._to_str = {i: t for t, i in self._to_id.items()}
        self.vocab_size = len(vocab)

    def encode(self, text, add_special_tokens=False):
        out, i = [], 0
        while i < len(text):
            for w in self._words:
                if text.startswith(w, i):
                    out.append(self._to_id[w])
                    i += len(w)
                    break
            else:
                out.append(self._to_id[text[i]])
                i += 1
        return out

    def decode(self, ids):
        return "".join(self._to_str[int(i)] for i in ids)


def _tool(name):
    return {"type": "function", "function": {
        "name": name,
        "parameters": {"type": "object",
                       "properties": {"location": {"type": "string"}},
                       "required": ["location"]}}}


def _logits(vocab, favored, value=10.0):
    a = [0.0] * vocab
    a[favored] = value
    return mx.array(a)[None]


def _simulate(grammar, tok, prompt_len=3, prefer=None, max_steps=200):
    """Run the processor greedily; ``prefer(step, generated)`` returns a token id
    the fake 'model' wants (defaults to JUNK, i.e. adversarial)."""
    junk = tok.encode(_JUNK)[0]
    proc = grammar.logits_processor()
    prompt = list(range(prompt_len))
    generated = []
    for step in range(max_steps):
        toks = mx.array(prompt + generated)
        want = junk if prefer is None else prefer(step, generated)
        logits = _logits(tok.vocab_size, want)
        masked = proc(toks, logits)
        nxt = int(mx.argmax(masked, axis=-1)[0])
        generated.append(nxt)
        # stop once the closing calls-end token has been emitted
        if grammar.calls_end and nxt == grammar.calls_end[-1] and step > 2:
            break
    return generated


def _prefer_from_text(tok, target):
    """A 'model' that wants ``target`` verbatim (used for the free JSON region)."""
    ids = tok.encode(target)
    return lambda step, gen: ids[step] if step < len(ids) else tok.encode(_JUNK)[0]


PASS, FAIL = [], []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def test_roundtrip_forced_despite_garbage():
    """tool_choice=required: structure is forced even though the model only ever
    wants JUNK in structural positions; JSON region follows the model."""
    tok = FakeTokenizer(["get_weather", "location", "Paris"])
    target = ('<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>'
              'get_weather\n```json\n{"location": "Paris"}\n```'
              '<｜tool▁call▁end｜><｜tool▁calls▁end｜>')
    full_ids = tok.encode(target)
    # json body token span within target
    pre = target.index("\n```json\n") + len("\n```json\n")
    j0 = len(tok.encode(target[:pre]))
    j1 = len(tok.encode(target[:pre + len('{"location": "Paris"}')]))
    want_ids = full_ids

    def prefer(step, gen):
        # adversarial everywhere except the free JSON body, where the model is competent
        return want_ids[step] if j0 <= step < j1 and step < len(want_ids) else tok.encode(_JUNK)[0]

    g = ToolCallGrammar(tok, [_tool("get_weather")], tool_choice="required")
    out = _simulate(g, tok, prefer=prefer)
    text = tok.decode(out)
    # The server strips the outer calls_begin/end; emulate by handing the whole
    # region to the parser (it ignores the outer wrapper via its inner regex).
    parsed = parse_tool_call(text)
    check("required: output token stream == canonical grammar", out == want_ids)
    check("required: parses to one call", isinstance(parsed, list) and len(parsed) == 1)
    check("required: name == get_weather", parsed and parsed[0]["name"] == "get_weather")
    check("required: arguments == {location: Paris}",
          parsed and parsed[0]["arguments"] == {"location": "Paris"})


def test_first_token_forced():
    """At step 0 in force mode, only calls_begin[0] survives — junk is masked."""
    tok = FakeTokenizer(["get_weather"])
    g = ToolCallGrammar(tok, [_tool("get_weather")], tool_choice="required")
    proc = g.logits_processor()
    junk = tok.encode(_JUNK)[0]
    masked = proc(mx.array([0, 1, 2]), _logits(tok.vocab_size, junk))
    nxt = int(mx.argmax(masked, axis=-1)[0])
    check("required: first token forced to calls_begin", nxt == g.calls_begin[0])
    check("required: junk logit driven to -inf",
          float(masked[0, junk]) < -1e8)


def test_name_constrained_to_tool_set():
    """In the name phase only registered-tool name tokens are allowed."""
    tok = FakeTokenizer(["get_weather", "search_web"])
    g = ToolCallGrammar(tok, [_tool("get_weather"), _tool("search_web")],
                        tool_choice="required")
    proc = g.logits_processor()
    # advance through calls_begin + call_head by feeding those exact tokens
    seq = list(g.calls_begin) + list(g.call_head)
    prompt = [0, 1, 2]
    for k in range(len(seq) + 1):
        toks = mx.array(prompt + seq[:k])
        masked = proc(toks, _logits(tok.vocab_size, tok.encode(_JUNK)[0]))
    # now in name phase: allowed first-tokens are the first token of each name
    allowed_first = {g.name_token_seqs[0][0], g.name_token_seqs[1][0]}
    row = masked[0]
    kept = {i for i in range(tok.vocab_size) if float(row[i]) > -1e8}
    check("name: only tool-name first-tokens allowed", kept == allowed_first)
    check("name: junk not allowed", tok.encode(_JUNK)[0] not in kept)


def test_auto_steps_aside_to_prose():
    """auto mode: if the model does not emit calls_begin at step 0, the processor
    stops constraining (prose)."""
    tok = FakeTokenizer(["get_weather"])
    g = ToolCallGrammar(tok, [_tool("get_weather")], tool_choice="auto")
    proc = g.logits_processor()
    junk = tok.encode(_JUNK)[0]
    # step 0: unconstrained (auto lets the model choose)
    m0 = proc(mx.array([0, 1, 2]), _logits(tok.vocab_size, junk))
    check("auto: step 0 unconstrained", float(m0[0, junk]) > -1e8)
    # model emitted junk (not calls_begin) -> prose thereafter
    m1 = proc(mx.array([0, 1, 2, junk]), _logits(tok.vocab_size, junk))
    check("auto: prose after non-tool first token (no masking)",
          all(float(m1[0, i]) > -1e8 for i in (junk, g.calls_begin[0])))


def test_json_completion_detector():
    jc = _JsonCompletion()
    for ch in '{"a": [1, 2], "b": "x}"}':
        jc.feed(ch)
    check("json: object with nested array + brace-in-string completes", jc.done)
    jc2 = _JsonCompletion()
    jc2.feed('{"a": 1')
    check("json: unbalanced object not complete", not jc2.done)


def main():
    for t in (test_roundtrip_forced_despite_garbage, test_first_token_forced,
              test_name_constrained_to_tool_set, test_auto_steps_aside_to_prose,
              test_json_completion_detector):
        print(t.__name__)
        t()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILURES:", FAIL)
        sys.exit(1)


if __name__ == "__main__":
    main()
