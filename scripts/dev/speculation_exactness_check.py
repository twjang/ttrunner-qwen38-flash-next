"""Does a speculative round emit what plain stepping emits?

    uv run python scripts/dev/speculation_exactness_check.py [tokens] [k]
                                                              (default 48, 4)

`docs/iterations/016` records that speculation is **not** exact even with greedy
acceptance -- "the verifier batches k rows where the stepper runs one, bf16
rounding differs in the last bits, and argmax amplifies it" -- with divergence
measured at tokens 29 and 39. Two things since then sit badly with that:

  * `step_n_check.py` finds `step_n` reproducing k sequential steps *exactly*
    at every k up to 32 -- 0.00 % on the hidden, tokens matching;
  * one row tile is the boundary where accumulation changes (invariant 13), and
    k <= 32 rows and 1 row are both inside it, so they should agree.

Speculation caps its widths at 17, so the whole scheme runs inside that tile.
This settles it by driving the round loop directly -- draft, snapshot, verify,
accept, restore, replay -- against a plain sequential decode from the same
start, in one process, all eager. No second trace, so no hang: this does not
need the defect in 5.6 fixed.
"""
import sys

import ttnn

from _device_model import open_model, synthetic_prompt

from twtest.tt.engine import accepted_prefix, prompt_lookup_draft

N = int(sys.argv[1]) if len(sys.argv) > 1 else 48
K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
PRE = 24                       # long enough for the n-gram drafter to find hits

mesh, cfg, m = open_model(max_seq_len=512)
# A prompt that repeats, so `prompt_lookup_draft` actually fires. Open prose
# never drafts and the comparison would be vacuous.
base = synthetic_prompt(8)
prompt = (base * 3)[:PRE]


def plain(n):
    st = m.new_state(batch=1)
    for t in prompt[:-1]:
        m.step([t], st)
    tok = m.greedy_tokens(m.step([prompt[-1]], st))[0]
    out = []
    for _ in range(n):
        out.append(int(tok))
        tok = m.greedy_tokens(m.step([int(tok)], st))[0]
    return out


def speculative(n):
    st = m.new_state(batch=1)
    for t in prompt[:-1]:
        m.step([t], st)
    tok = int(m.greedy_tokens(m.step([prompt[-1]], st))[0])
    out, rounds, drafted_rounds, accepted = [], 0, 0, 0
    while len(out) < n:
        rounds += 1
        context = list(st.histories[0]) + [tok]
        draft = prompt_lookup_draft(context, K - 1)
        if draft is None:                       # plain round
            out.append(tok)
            tok = int(m.greedy_tokens(m.step([tok], st))[0])
            continue
        drafted_rounds += 1
        feed = [tok] + draft
        snap = m.snapshot(st)
        verified = [int(x) for x in m.greedy_tokens(m.step_n(feed, st))[: len(feed)]]
        j = accepted_prefix(draft, verified[: len(feed) - 1])
        accepted += j
        emitted = verified[: j + 1]
        if j < len(draft):                      # the state ran past what we kept
            m.restore(st, snap)
            for t in feed[: j + 1]:
                m.step([t], st)
        out.extend(feed[: j + 1])
        tok = emitted[-1]
    return out[:n], rounds, drafted_rounds, accepted


ref = plain(N)
got, rounds, drafted_rounds, accepted = speculative(N)
same = ref == got
first = next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b), None)
print(f"RESULT k={K}  {rounds} rounds, {drafted_rounds} drafted, "
      f"{accepted} drafted tokens accepted", flush=True)
print(f"RESULT plain       {ref[:16]}", flush=True)
print(f"RESULT speculative {got[:16]}", flush=True)
print(f"RESULT {'IDENTICAL' if same else f'DIVERGES at token {first}'} "
      f"over {N} tokens", flush=True)
ttnn.close_mesh_device(mesh)
