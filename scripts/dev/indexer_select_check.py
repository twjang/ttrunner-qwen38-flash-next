"""Does the device select the tokens the reference's QSA mask allows?

    uv run python scripts/dev/indexer_select_check.py [tokens] [layer]
                                                       (default 2600, 3)

The selection *is* the feature, so this compares it directly and avoids the
thing that makes the obvious test impossible: a reference forward over 2600
tokens would take hours, so nothing here runs the model. Both sides are driven
from the same random hidden states -- the device one position at a time through
`_indexer_select`, the reference in one shot through `_indexer_mask` -- with the
same weights, so their selections have to agree.

Past `indexer_budget` there are more eligible blocks than the budget keeps, which
is the only regime where selection does anything: below it every complete block
is retained and dense attention is exactly right.
"""
import sys

import torch
import ttnn

from _device_model import host_row, open_model

from ttrunner_qwen38_flash_next.reference.cache import HybridCache
from ttrunner_qwen38_flash_next.reference.model import Qwen4ExpModel
from ttrunner_qwen38_flash_next.tt.model import LayerState

T = int(sys.argv[1]) if len(sys.argv) > 1 else 2600
LAYER = int(sys.argv[2]) if len(sys.argv) > 2 else 3

mesh, cfg, m = open_model(max_seq_len=max(4096, ((T + 127) // 128) * 128))
assert m.use_indexer, "indexer is off; max_seq_len must exceed the budget"
assert cfg.is_full_attention(LAYER)
ref = Qwen4ExpModel(cfg, m.host)

torch.manual_seed(0)
hidden = torch.randn(1, T, cfg.hidden_size) * 0.5

# -- reference: one call, the whole sequence ------------------------------
cache = HybridCache(cfg.num_layers)
positions = cache.extend_positions(torch.arange(T)[None])
cos, sin = ref.rotary(positions)
allowed = ref._indexer_mask(hidden.float(), LAYER, cos, sin, None, T, 0)[0, 0]   # (T, T)
print(f"RESULT reference mask built, {T} positions", flush=True)

# -- device: one position at a time --------------------------------------
st = LayerState()
for t in range(T):
    row = m.to_dev(hidden[:, t : t + 1].reshape(1, 1, 1, cfg.hidden_size).contiguous())
    # `_indexer_select` takes the query's rope table rather than deriving it:
    # `_attention_step` has already built cos/sin for the position and the
    # indexer reuses them. Same tables, same layout as the call site.
    q_cos, q_sin = m.rope([t])
    # `_indexer_select` returns (mask, page_table, cur_pos); the last two are
    # None unless compact attention is on, and this check opens the model well
    # below the length that turns it on.
    mask, _sel_table, _sel_pos = m._indexer_select(
        row, LAYER, st, [t], m.to_dev(q_cos, ttnn.float32), m.to_dev(q_sin, ttnn.float32)
    )
    if t % 500 == 0:
        print(f"  step {t}/{T}", flush=True)

# the mask is additive over the whole cache: 0 where the query may attend
msk = host_row(mesh, mask)[0, 0, 0].reshape(-1)
device_set = set(torch.nonzero(msk > -1e6).reshape(-1).tolist())
ref_set = set(torch.nonzero(allowed[T - 1]).reshape(-1).tolist())

print(f"RESULT position {T - 1}", flush=True)
print(f"RESULT device selected {len(device_set)} tokens, reference allows {len(ref_set)}", flush=True)
print(f"RESULT missing from device {len(ref_set - device_set)}   "
      f"extra on device {len(device_set - ref_set)}", flush=True)
inter = len(device_set & ref_set)
print(f"RESULT overlap {inter}/{len(ref_set)} = "
      f"{100 * inter / max(len(ref_set), 1):.1f}%", flush=True)
tail_lo = ((T - 1) + 1) // cfg.indexer_compress_ratio * cfg.indexer_compress_ratio
print(f"RESULT tail [{tail_lo}, {T - 1}] present on device: "
      f"{all(x in device_set for x in range(tail_lo, T))}", flush=True)

# A disagreement is only meaningful if the blocks were not tied. Recompute the
# reference's scores for the last query and report where the two selections
# parted, relative to the score at the selection boundary.
import math

import torch.nn.functional as F

from ttrunner_qwen38_flash_next.reference.layers import apply_rotary, rms_norm

ratio, dim = cfg.indexer_compress_ratio, cfg.indexer_head_dim
q = F.linear(hidden.float(), m.host.get(f"blk.{LAYER}.indexer.q_proj.weight").float())
q = rms_norm(q.reshape(1, T, -1, dim), m.host.get(f"blk.{LAYER}.indexer.q_norm.weight").float(),
             cfg.rms_norm_eps)
q = apply_rotary(q, cos, sin, unsqueeze_dim=2)[0, T - 1]                      # (heads, dim)
raw_k = F.linear(hidden.float(), m.host.get(f"blk.{LAYER}.indexer.k_proj.weight").float())
n_blocks = T // ratio
pooled = raw_k[0, : n_blocks * ratio].reshape(n_blocks, ratio, dim).mean(dim=1)
pooled = rms_norm(pooled, m.host.get(f"blk.{LAYER}.indexer.k_norm.weight").float(), cfg.rms_norm_eps)
starts = torch.arange(n_blocks) * ratio
pooled = apply_rotary(pooled[None, :, None], cos[:, starts], sin[:, starts], unsqueeze_dim=2)[0, :, 0]
sc = (torch.relu(q @ pooled.T).sum(dim=0) / math.sqrt(dim))                  # (n_blocks,)

def blocks_of(tokens):
    return {t // ratio for t in tokens if t < n_blocks * ratio}

miss = sorted(blocks_of(ref_set - device_set))
extra = sorted(blocks_of(device_set - ref_set))
order = sc.argsort(descending=True)
rank = {int(b): i for i, b in enumerate(order.tolist())}
cut = float(sc[order[cfg.indexer_budget // ratio - 1]])
print(f"RESULT blocks missing {miss} extra {extra}", flush=True)
for tag, bs in (("missing", miss), ("extra", extra)):
    for b in bs:
        print(f"RESULT   {tag} block {b}: score {float(sc[b]):.6f}, rank {rank[b]}, "
              f"gap to the 512th {float(sc[b]) - cut:+.2e}", flush=True)
ttnn.close_mesh_device(mesh)
