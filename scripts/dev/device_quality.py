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

import ttrunner_qwen38_flash_next.tt.model as model_mod

argv = sys.argv[1:]
PREFILL = 0
SCORE_FROM = 0
MOE_CHUNK = None
# `--chunk N` sets the prefill chunk width. Above 128 it needs
# TTRUNNER_WIDE_PREFILL_CHUNK=1, and it is not a free choice: a wider chunk changes
# the row count every dense linear sees, which changes their blocking and so
# their rounding (`row_count_stability_check.py`). This is how that gets priced
# in next-token accuracy rather than argued about.
CHUNK = 0
if "--chunk" in argv:
    i = argv.index("--chunk")
    CHUNK = int(argv[i + 1])
    argv = argv[:i] + argv[i + 2:]
if "--moe-chunk" in argv:
    i = argv.index("--moe-chunk")
    MOE_CHUNK = int(argv[i + 1])
    argv = argv[:i] + argv[i + 2:]
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
    "hieroglyphs, a writing system that had been unreadable for centuries. "
    "The stone was found in 1799 by French soldiers rebuilding a fort near the "
    "town of Rashid in the Nile Delta, and passed to British hands under the "
    "Capitulation of Alexandria two years later. Thomas Young established that "
    "the cartouches spelled a royal name phonetically, and Champollion showed "
    "that the hieroglyphic script recorded sounds as well as meanings, reading "
    "the name of Ramesses in 1822. The stele is a fragment of a larger stone "
    "whose missing upper portion probably carried a winged disc, and copies of "
    "the same decree have since been found at other sites, which is how the "
    "damaged passages of the Greek text were eventually restored."
)  # long enough for a full unpadded 128-token chunk

SEQ = int(os.environ.get("TTRUNNER_MAX_SEQ", "512"))
mesh, cfg, m = open_model(max_seq_len=SEQ)
print(f"RESULT max_seq_len {SEQ}  indexer {'on' if m.use_indexer else 'off'}", flush=True)
tok = tokenizer(cfg)
ids = tok.encode(TEXT)[:N]

st = m.new_state(batch=1)
hits = top5 = 0
nll = []
start = 0
if PREFILL:
    if MOE_CHUNK is not None and MOE_CHUNK > model_mod._MAX_MOE_CHUNK:
        # Lifting the cap is the point of asking: past 32 the MoE's answer moves,
        # and whether it moves *enough to matter* is exactly what this measures.
        print(f"RESULT lifting _MAX_MOE_CHUNK {model_mod._MAX_MOE_CHUNK} -> {MOE_CHUNK} "
              "for this measurement", flush=True)
        model_mod._MAX_MOE_CHUNK = MOE_CHUNK
    kw = {} if MOE_CHUNK is None else {"moe_chunk": MOE_CHUNK}
    if CHUNK:
        kw["chunk"] = CHUNK
    h = m.prefill(ids[:PREFILL], st, **kw)
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
