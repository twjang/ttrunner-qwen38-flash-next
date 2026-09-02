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

Then price it honestly, which for a recurrent model means paying for the
rollback. `step_n` advances the state by all k tokens; if only j are accepted the
DeltaNet recurrent state and the convolution rings are ahead by k-j and there is
no truncating them the way a transformer truncates its K/V cache. So a partial
round costs a snapshot, the verify, a restore, and a replay of the accepted
prefix:

    drafter fires, all k accepted    T(k)                       -> k+1 tokens
    drafter fires, j < k accepted    T(k) + T(j+1) + 2*SNAP     -> j+1 tokens
    drafter does not fire            S                          -> 1 token

Ignoring the replay is what makes speculation look better than it is: it is the
dominant correction here.
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


# measured on this machine: step_n_bench.py (eager) and
# traced_step_n_check.py (captured). The traced figures are the ones that
# matter -- an untraced verifier is beaten by the traced step it replaces.
STEP_EAGER, STEP_TRACED = 517.8, 235.9
T_EAGER = {1: 531.5, 2: 672.3, 4: 864.4, 8: 1233.1, 16: 2000.4}
T_TRACED = {1: 235.9, 2: 255.0, 3: 265.0, 4: 276.6, 5: 288.0, 6: 300.0,
            7: 312.0, 8: 323.9, 9: 335.0}
# snapshot/restore of the recurrent state: 36 layers x [12,1,128,128] float32
# is ~85 MB, at the ~35 GB/s the state matmuls see
SNAP = 2.5


def draft(context, ngram, k):
    """The k tokens that followed the last earlier occurrence of the last
    `ngram` of `context`, or None."""
    if len(context) <= ngram:
        return None
    key = context[-ngram:]
    for start in range(len(context) - ngram - 1, -1, -1):
        if context[start : start + ngram] == key:
            got = context[start + ngram : start + ngram + k]
            if len(got) == k:
                return got
            # too close to the end to propose k -- keep looking further back
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
            cost_t = cost_e = 0.0
            while pos < len(full) - 1:
                d = draft(full[:pos], ngram, k)
                if d is None:
                    plain += 1
                    emitted += 1
                    pos += 1
                    cost_t += STEP_TRACED
                    cost_e += STEP_EAGER
                    continue
                drafted += 1
                j = 0
                while j < k and pos + j < len(full) and full[pos + j] == d[j]:
                    j += 1
                accepted += j
                emitted += j + 1
                pos += j + 1
                cost_t += T_TRACED[k]
                cost_e += T_EAGER[k]
                if j < k:                      # roll back and replay the prefix
                    cost_t += T_TRACED[min(j + 1, max(T_TRACED))] + 2 * SNAP
                    cost_e += T_EAGER.get(min(j + 1, 16), T_EAGER[16]) + 2 * SNAP
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
