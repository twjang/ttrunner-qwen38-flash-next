"""Would n-gram drafting actually pay, given what `step_n` costs?

    uv run python scripts/dev/ngram_accept_rate.py [n_tokens]      (default 96)

Speculation's whole value is the acceptance rate, and that can be measured
without building any of it: generate greedily once, then run the drafter offline
against the token stream the model actually produced. A draft is accepted while
it agrees with that stream, which is exactly the greedy accept rule -- so the
counts here are the real ones, not an estimate.

Prompt-lookup drafting: find the most recent earlier occurrence of the last
`ngram` tokens and propose the `k` tokens that followed it. No draft model, no
extra weights.

Then price it. With `step_n` at k costing `T(k)` and a single step costing `S`:
a round where the drafter fires costs `T(k)` and emits j+1 tokens; a round where
it does not fire costs `S` and emits one. Charging `T(k)` for *every* round --
which the first version of this did -- understates the scheme badly, because the
drafter fires on a small minority of positions.
"""
import sys

import ttnn

from _device_model import open_model, tokenizer

N = int(sys.argv[1]) if len(sys.argv) > 1 else 96

# Two workloads, because prompt-lookup drafting is a bet on repetition.
TEXTS = {
    "open prose": (
        "The Rosetta Stone is a granodiorite stele inscribed with three versions "
        "of a decree issued in Memphis in 196 BC. The top and middle texts are in "
        "Ancient Egyptian, using hieroglyphic and Demotic scripts, while the "
        "bottom is in Ancient Greek."
    ),
    # the shape prompt-lookup is for: the answer is mostly quoted from the prompt
    "copy-heavy": (
        "Document: The maintenance window runs from 02:00 to 04:00 UTC on the "
        "first Sunday of each month. During the window the write path is "
        "unavailable and reads are served from the replica set.\n\n"
        "Question: when does the maintenance window run and what happens to the "
        "write path?\n\nAnswer: the maintenance window runs from"
    ),
}

mesh, cfg, m = open_model(max_seq_len=1024)
tok = tokenizer(cfg)

def generate(prompt):
    st = m.new_state(batch=1)
    h = None
    for t in prompt:
        h = m.step([t], st)
    out = []
    for _ in range(N):
        nxt = m.greedy_tokens(h)[0]
        out.append(nxt)
        h = m.step([nxt], st)
    del st
    return out


# measured on this machine, scripts/dev/step_n_bench.py (eager)
STEP_EAGER, STEP_TRACED = 517.8, 236.1
T = {1: 531.5, 2: 672.3, 4: 864.4, 8: 1233.1, 16: 2000.4}


def draft(context, ngram, k):
    """The k tokens that followed the last earlier occurrence of the last
    `ngram` of `context`, or None."""
    if len(context) <= ngram:
        return None
    key = context[-ngram:]
    for start in range(len(context) - ngram - 1, -1, -1):
        if context[start : start + ngram] == key:
            got = context[start + ngram : start + ngram + k]
            return got if len(got) == k else None
    return None


for label, text in TEXTS.items():
    prompt = tok.encode(text)
    stream = generate(prompt)
    print(f"RESULT [{label}] {N} tokens after a {len(prompt)}-token prompt", flush=True)
    print(f"RESULT [{label}] continuation {tok.decode(stream[:30])!r}", flush=True)
    full = prompt + stream

    for ngram in (2, 3):
        for k in (2, 4, 8):
            pos, drafted, plain, emitted, accepted = len(prompt), 0, 0, 0, 0
            while pos < len(full) - 1:
                d = draft(full[:pos], ngram, k)
                if d is None:
                    plain += 1
                    emitted += 1
                    pos += 1
                    continue
                drafted += 1
                j = 0
                while j < k and pos + j < len(full) and full[pos + j] == d[j]:
                    j += 1
                accepted += j
                emitted += j + 1
                pos += j + 1
            # a drafted round costs T(k); a round with no draft is a plain step
            cost_e = drafted * T[k] + plain * STEP_EAGER
            cost_t = drafted * T[k] + plain * STEP_TRACED
            fires = 100 * drafted / max(drafted + plain, 1)
            print(
                f"RESULT [{label}] ngram={ngram} k={k}: drafter fired on "
                f"{fires:4.0f}% of rounds, accepted {accepted}/{drafted * k} "
                f"({100 * accepted / max(drafted * k, 1):3.0f}%) | "
                f"vs eager {emitted * STEP_EAGER / cost_e:4.2f}x   "
                f"vs traced {emitted * STEP_TRACED / cost_t:4.2f}x",
                flush=True,
            )
ttnn.close_mesh_device(mesh)
