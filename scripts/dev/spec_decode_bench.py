"""Does traced speculation actually beat the plain step? Measured, end to end.

    uv run python scripts/dev/spec_decode_bench.py [n_tokens] [k]

The pieces are all verified separately now: traced `step_n(8)` is 22.71 ms a
token (section 8), and a rejected step is bit-exactly inert given the accept
mask and the conv-ring rewind (8.2, 8.3, `accept_mask_identity.py`). What is not
verified is the thing that decides whether any of it ships -- how often the
drafter is right, on real text.

The round, with **one** capture width so the alternation hang stays out of reach:

1. draft k-1 tokens with `prompt_lookup_draft`
2. verify all k in one traced replay, accept mask all ones
3. count the accepted prefix j
4. if j+1 < k the state has run ahead, so restore and replay the *same* width
   with the mask `[1]*(j+1) + [0]*...`, which advances exactly j+1

Step 4 is why this might not pay: a partial acceptance costs two verifies for
j+1 tokens. Break-even against a 96.2 ms plain step is j+1 >= 4 at k=8. The
honest answer therefore depends entirely on the drafter, which is why this
measures token-by-token acceptance rather than assuming it.

Prints the plain-step baseline on the same prompt for comparison, and the
acceptance histogram, because a mean hides whether the win is a few long runs or
a steady drip.
"""
import collections
import sys
import time

import ttnn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _device_model import open_model, tokenizer                     # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
K = int(sys.argv[2]) if len(sys.argv) > 2 else 8

mesh, cfg, m = open_model(max_seq_len=2048)
from ttrunner_qwen38_flash_next.tt.engine import (                   # noqa: E402
    accepted_prefix, prompt_lookup_draft,
)
from ttrunner_qwen38_flash_next.tt.traced import TracedStepN         # noqa: E402

tok = tokenizer(cfg)
_enc = tok.encode(
    "The history of computing hardware covers the developments from early "
    "simple devices to aid calculation to modern day computers. The history of "
    "computing hardware covers the developments from early simple devices."
)
PROMPT = list(getattr(_enc, "ids", _enc))
print(f"RESULT prompt {len(PROMPT)} tokens, generating {N}, k={K}, "
      f"use_indexer={m.use_indexer}", flush=True)


def prime(state):
    """Consume the prompt, leaving the last token to be fed."""
    for t in PROMPT[:-1]:
        m.step([t], state)
    return PROMPT[-1]


def capture(state, factory):
    """Build a traced object without losing the primed state.

    Capturing consumes warm-up tokens -- `TracedStepN.__init__` runs two plain
    steps and two `step_n` calls -- so it dirties whatever state it is handed.
    `_reprefill` in the engine wraps this in snapshot/restore for the same
    reason, and an earlier version of this benchmark did not: both runs then
    decoded from a state polluted by 2 + 2k tokens of the warm-up id, and the
    speculative and plain outputs agreed on 0 of 64 tokens.
    """
    snap = m.snapshot(state)
    try:
        return factory()
    finally:
        m.restore(state, snap)


# --- baseline: the plain traced step ----------------------------------------
from ttrunner_qwen38_flash_next.tt.traced import TracedDecoder        # noqa: E402

state = m.new_state(batch=1)
nxt = prime(state)
dec = capture(state, lambda: TracedDecoder(m, state))
out = []
t0 = time.perf_counter()
for _ in range(N):
    h = dec.step([nxt])
    nxt = int(m.greedy_tokens(h)[0])
    out.append(nxt)
plain = 1000 * (time.perf_counter() - t0) / N
print(f"RESULT plain traced step: {plain:7.2f} ms a token", flush=True)
print(f"RESULT text: {tok.decode(out[:24])!r}", flush=True)
dec.release()
baseline_out = list(out)

# --- speculation ------------------------------------------------------------
state = m.new_state(batch=1)
nxt = prime(state)
ver = capture(state, lambda: TracedStepN(m, state, K))
snap_buf = {}
hist = collections.Counter()
rounds = drafted = drafted_short = 0
out = []
t0 = time.perf_counter()
while len(out) < N:
    context = list(state.histories[0]) + [nxt]
    draft = prompt_lookup_draft(context, K - 1)
    rounds += 1
    # No draft -> one masked verify that advances by exactly 1. This looks
    # wasteful (a full k=8 verify for one token, ~191 ms) and it is, but the
    # alternative is worse and was measured: padding the draft so every round
    # verifies "properly" turned those rounds into *two* verifies, because a
    # padded draft is always rejected and a rejection has to rewind. 67.53 ms a
    # token became 86.08. The masked-1 path is optimal exactly when j is known
    # in advance to be 0.
    if draft is None or len(draft) != K - 1:
        h = ver.step_n([nxt] * K, accept=[1.0] + [0.0] * (K - 1))
        nxt = int(m.greedy_tokens(h)[0])
        out.append(nxt)
        hist[0] += 1
        continue

    drafted += 1
    feed = [nxt] + draft
    snap = m.snapshot(state, into=snap_buf.get("s"))
    snap_buf["s"] = snap
    verified = m.greedy_tokens(ver.step_n(feed))[:K]
    j = accepted_prefix(draft, verified[: K - 1])
    hist[j] += 1
    if j + 1 < K:
        m.restore(state, snap)
        ver.step_n(feed, accept=[1.0] * (j + 1) + [0.0] * (K - j - 1))
    out.extend(verified[: j + 1])
    nxt = verified[j]
spec = 1000 * (time.perf_counter() - t0) / len(out)
print(f"RESULT speculative:       {spec:7.2f} ms a token "
      f"({plain / spec:.2f}x against the plain step)", flush=True)
print(f"RESULT rounds {rounds}, full drafts {drafted}, padded {drafted_short} "
      f"({100 * drafted / max(rounds, 1):.0f}% real), {len(out)} tokens", flush=True)
print(f"RESULT accepted-prefix histogram: "
      f"{dict(sorted(hist.items()))}", flush=True)
match = sum(1 for a, b in zip(out, baseline_out) if a == b)
print(f"RESULT agrees with the plain step on {match}/{min(len(out), len(baseline_out))} "
      f"tokens", flush=True)
print(f"RESULT text: {tok.decode(out[:24])!r}", flush=True)
mean_j = sum(j * n for j, n in hist.items()) / max(sum(hist.values()), 1)
print(f"RESULT mean accepted prefix {mean_j:.2f} of {K - 1} drafted, "
      f"{len(out) / max(rounds, 1):.2f} tokens a round", flush=True)
print(f"RESULT target is 32.6 ms a token", flush=True)

ver.release()
ttnn.close_mesh_device(mesh)
