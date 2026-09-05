"""Does the compact window select exactly the tokens the dense mask does?

    uv run python scripts/dev/compact_window_equivalence.py [context]

The compact path re-addresses the QSA selection: instead of a mask with one
column per cache position, SDPA gets a page table listing only the pages the
selection touches, and a mask over that window. It is a change of *addressing*
and must not be a change of *selection* -- so this decodes one step and compares
the two, position by position.

Both paths are available between `compact_len` (17408) and the dense mask's
uint16 reach (65536), which is what makes the comparison possible at all. Below
17408 the dense row is smaller and compact is off; above 65536 only compact can
run, and there is nothing to compare it against.

`_indexer_mask` has no side effects -- `_indexer_update` maintains the block
cache and is a separate call -- so the same state can be asked twice.
"""
import sys

import torch
import ttnn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _device_model import open_model                                 # noqa: E402

CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 32768
KV_BLOCK = 32

mesh, cfg, m = open_model(max_seq_len=CTX)
print(f"RESULT max_seq_len {CTX}: use_indexer {m.use_indexer}, "
      f"compact_attention {m.compact_attention}, compact_len {m.compact_len} "
      f"({m.compact_slots} slots)", flush=True)
if not (m.use_indexer and m.compact_attention):
    print("RESULT both paths are needed for this comparison; pick a context "
          "between 17408 and 65536", flush=True)
    ttnn.close_mesh_device(mesh)
    raise SystemExit(0)

m.selection_active = True
state = m.new_state(batch=1)
# Enough tokens that the selection is doing real work: past the 2048 budget the
# eligible blocks outnumber the 512 slots and `topk` has to choose.
N = min(CTX - 1, 3000)
print(f"RESULT priming {N} tokens", flush=True)
for i in range(N):
    m.step([1000 + (i % 97)], state)

LAYER = next(l for l in range(cfg.num_layers) if cfg.is_full_attention(l))
print(f"RESULT comparing on QSA layer {LAYER} at position {N}", flush=True)

st = state.layers[LAYER]
row = m.to_dev(torch.randn(1, 1, 1, cfg.hidden_size) * 0.02)
q_cos, q_sin = m.rope([N])
cos = m.to_dev(q_cos, ttnn.float32)
sin = m.to_dev(q_sin, ttnn.float32)


def host(t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1]


# --- compact ---------------------------------------------------------------
cmask, ctable, ccur = m._indexer_mask(row, LAYER, st, [N], cos, sin)
ttnn.synchronize_device(mesh)
cm = host(cmask)[0, 0].reshape(cfg.num_attention_heads, -1)[0]
tbl = host(ctable)[0].to(torch.int64)
cols = torch.nonzero(cm > -1e6).reshape(-1).tolist()
compact_set = {int(tbl[c // KV_BLOCK]) * KV_BLOCK + (c % KV_BLOCK) for c in cols}
print(f"RESULT compact: {len(cols)} mask columns of {cm.numel()} open, "
      f"{len(compact_set)} distinct positions", flush=True)
if len(cols) != len(compact_set):
    print("RESULT !! two open columns address the same position -- that is a "
          "double count", flush=True)

# --- dense, same state -----------------------------------------------------
m.compact_attention = False
m._mask_base = None                       # the two modes want different widths
dmask, dtable, dcur = m._indexer_mask(row, LAYER, st, [N], cos, sin)
ttnn.synchronize_device(mesh)
assert dtable is None and dcur is None, "dense mode should not build a page table"
dm = host(dmask)[0, 0].reshape(cfg.num_attention_heads, -1)[0]
dense_set = set(torch.nonzero(dm > -1e6).reshape(-1).tolist())
print(f"RESULT dense:   {len(dense_set)} distinct positions of {dm.numel()} columns",
      flush=True)

only_c = sorted(compact_set - dense_set)
only_d = sorted(dense_set - compact_set)
print(f"RESULT compact-only {len(only_c)}, dense-only {len(only_d)}", flush=True)
if only_c[:5]:
    print(f"RESULT   first compact-only: {only_c[:5]}", flush=True)
if only_d[:5]:
    print(f"RESULT   first dense-only:   {only_d[:5]}", flush=True)
print("RESULT " + ("IDENTICAL selection" if not only_c and not only_d
                   else "SELECTIONS DIFFER"), flush=True)
print(f"RESULT bytes SDPA reads a layer: dense {2*2*cfg.head_dim*(N+1)*2*2/2**20:.1f} MiB, "
      f"compact {2*2*cfg.head_dim*m.compact_len*2*2/2**20:.1f} MiB", flush=True)

ttnn.close_mesh_device(mesh)
