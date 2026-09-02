"""Do the ops the QSA indexer needs exist, at the sizes it needs? On hardware.

    uv run python scripts/dev/probe_indexer_ops.py

Cheaper than discovering a dead end half way through an implementation. What it
established (2026-09-02, recorded in docs/HANDOFF.md 5.3):

  topk k=512            works at n = 2048, 8192 and 65536 (full context)
  gather dim=-2         needs the index uint32 in TILE_LAYOUT, shaped like the
                        output; int32 and ROW_MAJOR both assert
  sdpa_decode attn_mask a dead end -- must carry Q's head count, and throws at
                        program build even then, so select-and-gather it is
  scatter               rejects int32/uint32, accepts uint16

Re-run it after a tt-metal upgrade; these are the constraints the design is
shaped around.
"""
import torch, ttnn
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
def probe(tag, fn):
    try:
        print(f"PROBE {tag}: {fn()}", flush=True)
    except Exception as e:
        print(f"PROBE {tag}: FAILED {type(e).__name__}: {str(e)[:150]}", flush=True)

b, n_kv, S, hd, BUDGET = 1, 2, 8192, 256, 2048
cache = ttnn.from_torch(torch.randn(b, n_kv, S, hd), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=mesh)
idx_t = torch.randint(0, S, (b, n_kv, BUDGET, hd))
for dt, lay in ((ttnn.uint32, ttnn.TILE_LAYOUT), (ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                (ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)):
    try:
        idx = ttnn.from_torch(idx_t, dtype=dt, layout=lay, device=mesh)
        probe(f"gather dim=-2 idx {dt} {lay}",
              lambda idx=idx: tuple(ttnn.gather(cache, dim=-2, index=idx).shape))
    except Exception as e:
        print(f"PROBE make idx {dt} {lay}: FAILED {str(e)[:110]}", flush=True)

# the mask route, with Q's head count
n_q = 24
q = ttnn.from_torch(torch.randn(1, b, n_q, hd), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
v = ttnn.from_torch(torch.randn(b, n_kv, S, hd), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
pos = ttnn.from_torch(torch.tensor([100], dtype=torch.int32), dtype=ttnn.int32,
                      layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
m = ttnn.from_torch(torch.zeros(b, 1, n_q, S), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
probe(f"sdpa_decode attn_mask [b,1,{n_q},S]",
      lambda: tuple(ttnn.transformer.scaled_dot_product_attention_decode(
          q, cache, v, is_causal=False, attn_mask=m, cur_pos_tensor=pos, scale=hd**-0.5).shape))

# scatter with a non-i32 index dtype
N = 8192
base = ttnn.from_torch(torch.zeros(1, 1, 1, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
src = ttnn.from_torch(torch.ones(1, 1, 1, 512), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
for dt in (ttnn.uint32, ttnn.uint16):
    try:
        idx = ttnn.from_torch(torch.randint(0, N, (1, 1, 1, 512)), dtype=dt,
                              layout=ttnn.TILE_LAYOUT, device=mesh)
        probe(f"scatter idx {dt}", lambda idx=idx: tuple(ttnn.scatter(base, -1, idx, src).shape))
    except Exception as e:
        print(f"PROBE make scatter idx {dt}: FAILED {str(e)[:110]}", flush=True)
ttnn.close_mesh_device(mesh)
