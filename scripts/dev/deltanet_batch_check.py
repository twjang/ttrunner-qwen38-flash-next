"""Is batching the DeltaNet scan across chunks identical, and is it faster?

    uv run python scripts/dev/deltanet_batch_check.py [tokens] [reps]

`prefill(deltanet_batch=N)` runs N chunks as a group. Everything that depends on
the row count still sees one chunk's rows; only `prepare_device` and
`gated_delta_attn_seq` are batched, over their chunk axis. So the answer should
not move at all -- and `prepare_device` is 158.9 ms of a 922 ms chunk while
issuing ~59 dispatches whatever NC is, so on a dispatch-bound path most of that
should come back.

Checks both, and checks the *state* as well as the output: an output can be
right while the state left behind is not (invariant 18d). The state is compared
by taking eight decode steps afterwards and requiring the same token ids.
"""
import statistics
import sys
import time

import torch
import ttnn

from _device_model import open_model

TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 512
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
BATCHES = (1, 2, 4)

mesh, cfg, m = open_model(max_seq_len=1024)
torch.manual_seed(0)
prompt = [(1000 + 7 * i) % 30000 for i in range(TOKENS)]


def run(batch, chunk=None):
    st = m.new_state(batch=1)
    kw = {"deltanet_batch": batch}
    if chunk:
        kw["chunk"] = chunk
    h = m.prefill(prompt, st, **kw)
    logits = m.logits(h)[0].float().clone()
    toks = []
    for _ in range(8):
        t = int(logits.argmax())
        toks.append(t)
        logits = m.logits(m.step([t], st))[0].float()
    return logits, toks, h


def timed(batch):
    st = m.new_state(batch=1)
    t0 = time.perf_counter()
    m.prefill(prompt, st, deltanet_batch=batch)
    ttnn.synchronize_device(mesh)
    return 1000 * (time.perf_counter() - t0)


ref_logits, ref_toks, _ = run(1)
print(f"RESULT batch=1 reference: first 8 tokens {ref_toks}", flush=True)
for b in BATCHES[1:]:
    lg, toks, _ = run(b)
    same_state = toks == ref_toks
    d = (lg - ref_logits).abs()
    print(f"RESULT batch={b}: continuation identical {'YES' if same_state else 'NO'}   "
          f"logit max diff after 8 steps {d.max().item():.3e}", flush=True)
    if not same_state:
        print(f"RESULT   batch={b} tokens {toks}", flush=True)

for b in BATCHES:
    timed(b)
    ts = sorted(timed(b) for _ in range(REPS))
    med = statistics.median(ts)
    print(f"RESULT batch={b}: prefill {TOKENS} tokens median {med:7.1f} ms  "
          f"min {ts[0]:7.1f}   {med / TOKENS:.2f} ms/token", flush=True)

# And the other axis: chunk width. This one is *not* bit-identical -- five of the
# ten dense linear shapes change blocking above 128 rows -- so the check is that
# the difference is the size of bf16 rounding rather than a bug, and that the
# continuation still tracks.
lg128, toks128, _ = run(1, chunk=128)
lg512, toks512, _ = run(1, chunk=512)
d = (lg512 - lg128).abs()
agree = sum(int(a == b) for a, b in zip(toks128, toks512))
print(f"RESULT chunk 128 vs 512: logit max diff {d.max().item():.3e}  "
      f"mean {d.mean().item():.3e}  first-8 tokens agree {agree}/8", flush=True)
print(f"RESULT   chunk128 {toks128}", flush=True)
print(f"RESULT   chunk512 {toks512}", flush=True)

ttnn.close_mesh_device(mesh)
