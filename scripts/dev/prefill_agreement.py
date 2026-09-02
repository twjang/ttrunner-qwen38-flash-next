"""Does a prompt consumed by `prefill` predict what the decode path predicts?

    uv run python scripts/dev/prefill_agreement.py [prefill_len] [tail_len]
                                                    (default 128, 32)

`prefill_bisect.py` and `decode_vs_reference.py` compare hidden states, and that
turns out not to discriminate: the *verified* decode path sits ~50 % away from
the float32 reference at every layer, flat from layer 0, which is what discrete
MoE routing divergence looks like (top-10 of 512 experts, bfloat4_b router --
one different expert changes a hidden state a lot and the token not at all). A
hidden-state threshold therefore cannot tell a prefill bug from a routing coin
flip.

So ask the question the product actually cares about: after prefill, does the
state predict the same tokens? Both paths are teacher-forced through the same
tail of the prompt, so disagreements are counted independently instead of
compounding after the first one.

Two prompts: a real sentence, where the decode path's continuation is known
good (' Paris.\\n\\nThe French city in Europe'), and a synthetic one long
enough to need more than one prefill chunk.
"""
import sys

import ttnn

from _device_model import open_model, synthetic_prompt, tokenizer

PRE = int(sys.argv[1]) if len(sys.argv) > 1 else 128
TAIL = int(sys.argv[2]) if len(sys.argv) > 2 else 32

mesh, cfg, m = open_model(max_seq_len=max(512, PRE + TAIL + 64))
tok = tokenizer(cfg)


def step_argmaxes(prompt, tail):
    """Feed prompt+tail one token at a time; return the argmax after each tail token."""
    st = m.new_state(batch=1)
    for t in prompt:
        m.step([t], st)
    out = []
    for t in tail:
        out.append(m.greedy_tokens(m.step([t], st))[0])
    del st
    return out


def prefill_argmaxes(prompt, tail):
    st = m.new_state(batch=1)
    m.prefill(prompt, st)
    out = []
    for t in tail:
        out.append(m.greedy_tokens(m.step([t], st))[0])
    del st
    return out


def generate(prompt, n, use_prefill):
    st = m.new_state(batch=1)
    if use_prefill:
        h = m.prefill(prompt, st)
    else:
        for t in prompt:
            h = m.step([t], st)
    out = []
    for _ in range(n):
        t = m.greedy_tokens(h)[0]
        out.append(t)
        h = m.step([t], st)
    del st
    return out


# -- 1. the known-good sentence -------------------------------------------
real = tok.encode("The capital of France is")
g_step = generate(real, 8, use_prefill=False)
g_pre = generate(real, 8, use_prefill=True)
print(f"RESULT real prompt {len(real)} tokens", flush=True)
print(f"RESULT   step    {g_step} {tok.decode(g_step)!r}", flush=True)
print(f"RESULT   prefill {g_pre} {tok.decode(g_pre)!r}", flush=True)
print(f"RESULT   continuations {'MATCH' if g_step == g_pre else 'DIFFER'}", flush=True)

# -- 2. teacher-forced agreement over a long synthetic prompt -------------
full = synthetic_prompt(PRE + TAIL)
prompt, tail = full[:PRE], full[PRE:]
a = step_argmaxes(prompt, tail)
b = prefill_argmaxes(prompt, tail)
agree = sum(x == y for x, y in zip(a, b))
first_diff = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
print(f"RESULT synthetic prefill_len={PRE} tail={TAIL}", flush=True)
print(f"RESULT   agreement {agree}/{len(a)} = {100 * agree / len(a):.1f}%", flush=True)
print(f"RESULT   first disagreement at tail index {first_diff}", flush=True)

ttnn.close_mesh_device(mesh)
