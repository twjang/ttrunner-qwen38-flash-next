"""Oracle vs device decode vs device prefill, teacher-forced, on real text.

    uv run python scripts/dev/three_way_agreement.py [total] [pre]  (default 48 16)

The question prefill has to answer is not "does it match the decode path" but
"is it as good as the decode path". The decode path is itself a bf16 4-bit
approximation and disagrees with the float32 oracle on some tokens, so a
prefill/decode disagreement rate means nothing without that baseline.

Greedy agreement is a brittle metric for this model: 512 experts with top-10
routing means any perturbation can flip a selection, and a flipped selection can
change a token, so two implementations of the *same* weights disagree wherever
the margin is thin. So this also reports the negative log-likelihood each path
assigns the text's actual next token -- a continuous measure that says whether
the model is still the same model, which agreement alone cannot.

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
# Prefill step size. The DeltaNet op's error grows with position inside its
# 128-wide chunk, so a smaller step trades prefill throughput for accuracy.
CHUNKS = [int(x) for x in sys.argv[3:]] or [128]

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
oracle, margin, ref_nll = [], [], []
for i in range(TOTAL):
    lg = ref.logits(h[0, i]).float()
    top2 = lg.topk(2).values
    oracle.append(int(lg.argmax()))
    # How much the oracle itself prefers its answer. A disagreement where this
    # is tiny is a coin flip the device lost, not evidence of a defect.
    margin.append(float(top2[0] - top2[1]))
    if i + 1 < TOTAL:
        ref_nll.append(float(-torch.log_softmax(lg, dim=-1)[ids[i + 1]]))
# Is the oracle a sane language model on this text at all? If its own top-1
# accuracy against the actual next token is near zero, or its NLL is dominated
# by a handful of impossible positions, then "agreement with the oracle" is
# measuring against something broken and every number below is meaningless.
hits = sum(oracle[i] == ids[i + 1] for i in range(TOTAL - 1))
srt = sorted(ref_nll)
print(f"RESULT oracle ready, {TOTAL} positions", flush=True)
print(f"RESULT oracle predicts the actual next token {hits}/{TOTAL - 1} = "
      f"{100 * hits / (TOTAL - 1):.1f}%", flush=True)
print(f"RESULT oracle NLL  median {srt[len(srt) // 2]:.3f}  min {srt[0]:.3f}  "
      f"max {srt[-1]:.3f}", flush=True)
print(f"RESULT oracle NLL per position {[round(x, 1) for x in ref_nll]}", flush=True)

# -- device decode: a prediction after every token -------------------------
st = m.new_state(batch=1)
step_out, step_nll = [], []
for i, t in enumerate(ids):
    hs = m.step([t], st)
    step_out.append(m.greedy_tokens(hs)[0])
    if i + 1 < TOTAL:
        step_nll.append(float(-torch.log_softmax(m.logits(hs)[0].float(), dim=-1)[ids[i + 1]]))
del st

# -- device prefill for the first PRE, then decode -------------------------
prefills = {}
for c in CHUNKS:
    st = m.new_state(batch=1)
    hp = m.prefill(ids[:PRE], st, chunk=max(32, min(c, 128) // 32 * 32))
    out = [None] * PRE
    out[PRE - 1] = m.greedy_tokens(hp)[0]
    for t in ids[PRE:]:
        out.append(m.greedy_tokens(m.step([t], st))[0])
    prefills[c] = out
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


def nll(name, values, lo):
    idx = [i for i in range(lo, len(values)) if values[i] is not None]
    mean = sum(values[i] for i in idx) / len(idx)
    print(f"RESULT {name:8s} mean NLL {mean:.4f} over {len(idx)} positions "
          f"(perplexity {pow(2.718281828, mean):.2f})", flush=True)
    return mean


score("step", step_out, 1)
r = nll("oracle", ref_nll, 1)
d = nll("step", step_nll, 1)
print(f"RESULT step NLL is {d - r:+.4f} vs the oracle "
      f"({100 * (pow(2.718281828, d - r) - 1):+.1f} % perplexity)", flush=True)
idx = [i for i in range(PRE - 1, TOTAL)]
for c, out in prefills.items():
    score(f"pre/{c}", out, PRE - 1)
    same = sum(out[i] == step_out[i] for i in idx)
    print(f"RESULT pre/{c:<4d} vs step {same}/{len(idx)} = {100 * same / len(idx):.1f}%", flush=True)
ttnn.close_mesh_device(mesh)
