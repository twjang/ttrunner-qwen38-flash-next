"""Oracle vs device decode vs device prefill, teacher-forced, on real text.

    uv run python scripts/dev/three_way_agreement.py [pre] [tail]   (default 16 16)

The question prefill has to answer is not "does it match the decode path" but
"is it as good as the decode path". The decode path is itself a bf16 4-bit
approximation and disagrees with the float32 oracle on some tokens, so a
prefill/decode disagreement rate means nothing without that baseline.

One whole-prompt reference forward yields the oracle's argmax at *every*
position at once, which makes the baseline affordable. All three paths see the
same tokens, so disagreements are counted independently instead of compounding.
"""
import sys

import torch
import ttnn

from _device_model import open_model, tokenizer

from twtest.reference.cache import HybridCache
from twtest.reference.model import Qwen4ExpModel

PRE = int(sys.argv[1]) if len(sys.argv) > 1 else 16
TAIL = int(sys.argv[2]) if len(sys.argv) > 2 else 16

TEXT = (
    "The Rosetta Stone is a granodiorite stele inscribed with three versions of "
    "a decree issued in Memphis in 196 BC. The top and middle texts are in "
    "Ancient Egyptian, using hieroglyphic and Demotic scripts, while the bottom "
    "is in Ancient Greek. Because the decree has only minor differences between "
    "the three versions, the stone proved to be the key to deciphering Egyptian "
    "hieroglyphs, a writing system that had been unreadable for centuries."
)

mesh, cfg, m = open_model()
tok = tokenizer(cfg)
ids = tok.encode(TEXT)
if len(ids) < PRE + TAIL:
    raise SystemExit(f"text is only {len(ids)} tokens, need {PRE + TAIL}")
ids = ids[: PRE + TAIL]
prompt, tail = ids[:PRE], ids[PRE:]

ref = Qwen4ExpModel(cfg, m.host)
h = ref.forward(torch.tensor([ids]), HybridCache(cfg.num_layers))
oracle = [int(ref.logits(h[0, PRE - 1 + i]).argmax()) for i in range(TAIL)]
print(f"RESULT oracle {oracle}", flush=True)

st = m.new_state(batch=1)
for t in prompt:
    hs = m.step([t], st)
step_out = [m.greedy_tokens(hs)[0]]
for t in tail[:-1]:
    step_out.append(m.greedy_tokens(m.step([t], st))[0])
del st
print(f"RESULT step   {step_out}", flush=True)

st = m.new_state(batch=1)
hp = m.prefill(prompt, st)
pre_out = [m.greedy_tokens(hp)[0]]
for t in tail[:-1]:
    pre_out.append(m.greedy_tokens(m.step([t], st))[0])
del st
print(f"RESULT prefill {pre_out}", flush=True)


def agree(a, b):
    n = sum(x == y for x, y in zip(a, b))
    return f"{n}/{len(a)} = {100 * n / len(a):.1f}%"


print(f"RESULT step vs oracle    {agree(step_out, oracle)}", flush=True)
print(f"RESULT prefill vs oracle {agree(pre_out, oracle)}", flush=True)
print(f"RESULT prefill vs step   {agree(pre_out, step_out)}", flush=True)
ttnn.close_mesh_device(mesh)
