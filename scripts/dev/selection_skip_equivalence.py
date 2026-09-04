"""Is skipping the QSA selection below the budget actually exact?

    uv run python scripts/dev/selection_skip_equivalence.py [n_tokens]

Handoff 4f claims it is, on arithmetic: eligible blocks at position p number
`p // ratio`, so while `p < indexer_budget` there are fewer than `indexer_topk`
of them, `topk` returns every visible block, and the mask it builds is exactly
plain causal attention. `tests/test_indexer_selection.py` pins that arithmetic,
but arithmetic about the mask is not the same as the model agreeing token for
token, and the engine now ships with the selection off by default.

So: open the model large enough that the indexer engages at all (max_seq_len
above the budget, or `use_indexer` is False and this proves nothing -- the run
asserts it), greedily decode the same prompt twice, once with the selection
running and once skipped, and compare.

Exact means exact: identical token ids, and the hidden states bitwise equal. A
difference at any position below the budget would mean the engine's fast path
changes the answer, whatever the arithmetic says.
"""
import sys

import torch
import ttnn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _device_model import open_model, tokenizer                     # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 32

mesh, cfg, m = open_model(max_seq_len=4096)
tok = tokenizer(cfg)

if not m.use_indexer:
    print("RESULT indexer is off at this size -- nothing to compare", flush=True)
    ttnn.close_mesh_device(mesh)
    raise SystemExit(1)
print(f"RESULT indexer on: budget={m.indexer_budget} ratio={m.indexer_ratio} "
      f"topk={m.indexer_topk} max_blocks={m.max_blocks}", flush=True)

_enc = tok.encode(
    "The history of computing hardware covers the developments from early "
    "simple devices to aid calculation to modern day computers."
)
prompt = list(getattr(_enc, "ids", _enc))
print(f"RESULT prompt {len(prompt)} tokens, generating {N}", flush=True)


def run(selection: bool):
    """Greedy-decode N tokens, returning the ids and the final hidden state."""
    m.selection_active = selection
    st = m.new_state(batch=1)
    hidden = None
    for t in prompt:
        hidden = m.step([t], st)
    out, hiddens = [], []
    for _ in range(N):
        # `greedy_tokens` already returns host ints -- one per sequence -- or
        # None when the vocabulary split is uneven, which it is not here.
        ids = m.greedy_tokens(hidden)
        if ids is None:
            ids = [int(m.from_dev(m.logits(hidden))[0].flatten().argmax())]
        nxt = int(ids[0])
        out.append(nxt)
        hidden = m.step([nxt], st)
        hiddens.append(m.from_dev(hidden).to(torch.float32).clone())
    return out, hiddens, max(st.positions)


on_ids, on_h, pos = run(True)
print(f"RESULT selection ON  final position {pos} (budget {m.indexer_budget})", flush=True)
off_ids, off_h, _ = run(False)
print("RESULT selection OFF done", flush=True)

if pos >= m.indexer_budget:
    print(f"RESULT WARNING ran past the budget ({pos} >= {m.indexer_budget}); "
          "the two are not expected to agree there", flush=True)

same_ids = on_ids == off_ids
first_diff = next((i for i, (a, b) in enumerate(zip(on_ids, off_ids)) if a != b), None)
worst = max((a - b).abs().max().item() for a, b in zip(on_h, off_h))
bitwise = all(torch.equal(a, b) for a, b in zip(on_h, off_h))

print(f"RESULT token ids identical: {same_ids}"
      + ("" if same_ids else f" (first difference at step {first_diff})"), flush=True)
print(f"RESULT hidden states bitwise identical: {bitwise}  max abs diff {worst:.3e}",
      flush=True)
print("RESULT " + ("EXACT -- the skip changes nothing below the budget"
                   if same_ids and bitwise else
                   "NOT EXACT -- handoff 4f and invariant 36 are wrong"), flush=True)
print(f"RESULT on  {tok.decode(on_ids)!r}", flush=True)
if not same_ids:
    print(f"RESULT off {tok.decode(off_ids)!r}", flush=True)

ttnn.close_mesh_device(mesh)
