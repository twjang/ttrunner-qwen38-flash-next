"""Is the *device* a working language model? No oracle in the loop.

    uv run python scripts/dev/device_quality.py [n_tokens] [--prefill P]
                                                       (default 48, no prefill)

`--prefill P` consumes the first P tokens through the chunked prefill path and
steps the rest, scoring only the stepped positions -- so prefill is judged by
whether the state it leaves still predicts the text, not by whether it matches
the decode path token for token.

The counterpart of `reference_quality.py`, and the metric this project should
have been using all along. Agreement with the float32 reference turned out to be
dominated by chaos -- 512 experts with top-10 routing, so any perturbation flips
a selection somewhere -- and every precision configuration measured landed in the
same 30-47 % band. Next-token accuracy against the *actual text* has no such
problem: it is absolute, it needs no reference forward, and a broken model
cannot fake it (a damaged model still emits grammatical, repetitive English,
which is exactly how a wrong head pairing survived being eyeballed).

The float32 reference scores 80.9 % top-1 / 97.9 % top-5, mean NLL 0.703.
"""
import os
import sys

import torch
import ttnn

from _device_model import open_model, tokenizer

argv = sys.argv[1:]
PREFILL = 0
SCORE_FROM = 0
for flag in ("--prefill", "--score-from"):
    if flag in argv:
        i = argv.index(flag)
        if flag == "--prefill":
            PREFILL = int(argv[i + 1])
        else:
            SCORE_FROM = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
N = int(argv[0]) if argv else 48

TEXT = (
    "The Rosetta Stone is a granodiorite stele inscribed with three versions of "
    "a decree issued in Memphis in 196 BC. The top and middle texts are in "
    "Ancient Egyptian, using hieroglyphic and Demotic scripts, while the bottom "
    "is in Ancient Greek. Because the decree has only minor differences between "
    "the three versions, the stone proved to be the key to deciphering Egyptian "
    "hieroglyphs, a writing system that had been unreadable for centuries."
)

SEQ = int(os.environ.get("TWTEST_MAX_SEQ", "512"))
mesh, cfg, m = open_model(max_seq_len=SEQ)
print(f"RESULT max_seq_len {SEQ}  indexer {'on' if m.use_indexer else 'off'}", flush=True)
tok = tokenizer(cfg)
ids = tok.encode(TEXT)[:N]

st = m.new_state(batch=1)
hits = top5 = 0
nll = []
start = 0
if PREFILL:
    h = m.prefill(ids[:PREFILL], st)
    lg = m.logits(h)[0].float()
    target = ids[PREFILL]
    hits += int(lg.argmax()) == target
    top5 += target in lg.topk(5).indices.tolist()
    nll.append(float(-torch.log_softmax(lg, dim=-1)[target]))
    start = PREFILL
for i in range(start, len(ids) - 1):
    h = m.step([ids[i]], st)
    if i < SCORE_FROM:
        continue                      # warmed but not scored, for a fair control
    lg = m.logits(h)[0].float()
    target = ids[i + 1]
    hits += int(lg.argmax()) == target
    top5 += target in lg.topk(5).indices.tolist()
    nll.append(float(-torch.log_softmax(lg, dim=-1)[target]))
n = len(nll)
srt = sorted(nll)
print(f"RESULT device (prefill={PREFILL}, from={max(PREFILL, SCORE_FROM)}) next-token top-1 {hits}/{n} = {100 * hits / n:.1f}%   "
      f"top-5 {top5}/{n} = {100 * top5 / n:.1f}%", flush=True)
print(f"RESULT device NLL mean {sum(nll) / n:.3f}  median {srt[n // 2]:.3f}  "
      f"perplexity(mean) {pow(2.718281828, sum(nll) / n):.2f}", flush=True)
print("RESULT float32 reference: 80.9% top-1, 97.9% top-5, mean NLL 0.703", flush=True)
ttnn.close_mesh_device(mesh)
