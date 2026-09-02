"""Oracle vs device decode vs device prefill, teacher-forced, on real text.

    uv run python scripts/dev/three_way_agreement.py [total] [pre]  (default 48 16)

The question prefill has to answer is not "does it match the decode path" but
"is it as good as the decode path". The decode path is itself a bf16 4-bit
approximation and disagrees with the float32 oracle on some tokens, so a
prefill/decode disagreement rate means nothing without that baseline.

One whole-prompt reference forward yields the oracle's argmax at *every*
position at once, which makes the baseline affordable -- so every position is
used, not just a tail. All three paths see the same tokens, so disagreements are
counted independently instead of compounding. The decode path is scored over
positions 1..total-1 and prefill over pre..total-1, the ones it is responsible
for.
"""
import sys

import torch
import ttnn

from _device_model import open_model, tokenizer

from twtest.reference.cache import HybridCache
from twtest.reference.model import Qwen4ExpModel

TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 48
PRE = int(sys.argv[2]) if len(sys.argv) > 2 else 16

TEXT = (
    "The Rosetta Stone is a granodiorite stele inscribed with three versions of "
    "a decree issued in Memphis in 196 BC. The top and middle texts are in "
    "Ancient Egyptian, using hieroglyphic and Demotic scripts, while the bottom "
    "is in Ancient Greek. Because the decree has only minor differences between "
    "the three versions, the stone proved to be the key to deciphering Egyptian "
    "hieroglyphs, a writing system that had been unreadable for centuries."
)

mesh, cfg, m = open_model(max_seq_len=512)
tok = tokenizer(cfg)
ids = tok.encode(TEXT)
if len(ids) < TOTAL:
    raise SystemExit(f"text is only {len(ids)} tokens, need {TOTAL}")
ids = ids[:TOTAL]

# -- oracle: one forward, a prediction at every position -------------------
ref = Qwen4ExpModel(cfg, m.host)
h = ref.forward(torch.tensor([ids]), HybridCache(cfg.num_layers))
oracle, margin = [], []
for i in range(TOTAL):
    lg = ref.logits(h[0, i]).float()
    top2 = lg.topk(2).values
    oracle.append(int(lg.argmax()))
    # How much the oracle itself prefers its answer. A disagreement where this
    # is tiny is a coin flip the device lost, not evidence of a defect.
    margin.append(float(top2[0] - top2[1]))
print(f"RESULT oracle ready, {TOTAL} positions", flush=True)

# -- device decode: a prediction after every token -------------------------
st = m.new_state(batch=1)
step_out = [m.greedy_tokens(m.step([t], st))[0] for t in ids]
del st

# -- device prefill for the first PRE, then decode -------------------------
st = m.new_state(batch=1)
hp = m.prefill(ids[:PRE], st)
pre_out = [None] * PRE
pre_out[PRE - 1] = m.greedy_tokens(hp)[0]
for t in ids[PRE:]:
    pre_out.append(m.greedy_tokens(m.step([t], st))[0])
del st


def score(name, got, lo):
    idx = [i for i in range(lo, TOTAL) if got[i] is not None]
    n = sum(got[i] == oracle[i] for i in idx)
    print(f"RESULT {name:8s} overall {n}/{len(idx)} = {100 * n / len(idx):.1f}%", flush=True)
    for label, keep in (
        ("confident (margin > 2)", [i for i in idx if margin[i] > 2]),
        ("close     (margin <= 2)", [i for i in idx if margin[i] <= 2]),
    ):
        if not keep:
            continue
        k = sum(got[i] == oracle[i] for i in keep)
        print(f"RESULT {name:8s} {label} {k}/{len(keep)} = {100 * k / len(keep):.0f}%", flush=True)


score("step", step_out, 1)
score("prefill", pre_out, PRE - 1)
idx = [i for i in range(PRE - 1, TOTAL)]
same = sum(pre_out[i] == step_out[i] for i in idx)
print(f"RESULT prefill vs step {same}/{len(idx)} = {100 * same / len(idx):.1f}%", flush=True)
ttnn.close_mesh_device(mesh)
