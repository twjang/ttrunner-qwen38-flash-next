"""Does decode quality hold as the context grows? It does not.

    uv run python scripts/dev/context_decay_check.py [positions] [block]
                                                        (default 300, 32)

Every accuracy number this project has ever quoted was measured over at most 128
scored positions -- `device_quality.py` defaults to 48 and its passage is ~200
tokens long -- so nothing ever looked past the point where this starts. Stepping
one token at a time and reporting top-1 per block of positions:

    positions    0- 127   78-84 %
    positions  128- 159   53.1 %
    positions  160- 191   12.5 %
    positions  192- 223    6.2 %
    positions  224+        0.0 %

Zero top-1 on ordinary English is not "hard text": a working language model gets
the common tokens right whatever else it does. This is a defect.

What is established about it:

* It is **pure decode** -- one `m.step` per token, no prefill anywhere in the
  loop -- so it is not the chunked-prefill path.
* It **predates this session's MoE work**: reverting `moe.py` to 4a43a7a gives
  the same curve (78.1 / 53.1 / 9.4 / 3.1 / 0.0).
* It is **not tied to `max_seq_len`**: 512 and 2048 collapse at the same
  positions, so it is not the K/V page table running out.
* It reproduces on a second text through a second harness:
  `device_quality.py 200` scores 62.3 % where the same harness scores 80.3 % at
  N=128, which is the same collapse averaged in.

Not yet established: which component. The onset is at 128, which is `CHUNK` and
`cfg.linear_head_dim`, but decode uses neither -- it steps through
`_linear_attention_step`, so the DeltaNet recurrent state ([BH, 1, 128, 128]) and
the conv/PLE rings are the things that carry position. `decode_ablation_check.py`
has the seams to bisect it.
"""
import os
import sys

import ttnn

from _device_model import open_model, tokenizer

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 300
BLOCK = int(sys.argv[2]) if len(sys.argv) > 2 else 32

PARA = (
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
    "the name of Ramesses in 1822. "
)
EXTRA = (
    "Printing with movable type reached Europe in the fifteenth century, and the "
    "presses that followed changed what a book cost and therefore who could own "
    "one. Early editions imitated manuscript hands closely, because readers "
    "judged a page by the standards they already had. Roman types displaced "
    "blackletter in southern Europe within two generations, while northern "
    "printers kept the older forms for vernacular texts long after scholarly "
    "works had moved. Paper supply, not type, set the ceiling on an edition's "
    "size, and mills clustered where clean water and rags were both available. "
)
TEXT = PARA + EXTRA + PARA + EXTRA

SEQ = int(os.environ.get("TTRUNNER_MAX_SEQ", "512"))
mesh, cfg, m = open_model(max_seq_len=SEQ)
tok = tokenizer(cfg)
ids = tok.encode(TEXT)
if len(ids) < LIMIT + 1:
    raise SystemExit(f"RESULT text too short: {len(ids)} < {LIMIT + 1}")

print(f"RESULT max_seq_len {SEQ}  scoring {LIMIT} stepped positions", flush=True)
st = m.new_state(batch=1)
hits = tot = 0
lo = 0
for i in range(LIMIT):
    h = m.step([ids[i]], st)
    lg = m.logits(h)[0].float()
    hits += int(int(lg.argmax()) == ids[i + 1])
    tot += 1
    if tot == BLOCK or i == LIMIT - 1:
        print(f"RESULT positions {lo:4d}-{i:4d}: top-1 {hits:3d}/{tot:3d} = "
              f"{100 * hits / tot:5.1f}%", flush=True)
        hits = tot = 0
        lo = i + 1

ttnn.close_mesh_device(mesh)
