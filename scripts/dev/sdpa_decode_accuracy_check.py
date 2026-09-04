"""Is the paged decode attention right as the cache grows past one k-chunk?

    uv run python scripts/dev/sdpa_decode_accuracy_check.py [k_chunk]

Decode's next-token accuracy falls off from ~position 128 and reaches zero by
224 (handoff 4b), and the shape of the fall moves with `sdpa_k_chunk` -- 64
breaks from ~64, 128 from ~128, 256 decays gently from 128 and sharply at 256.
That points at the online softmax over multiple k-chunks, but pointing is not
proving: the model has a DeltaNet recurrence and a PLE ring that also carry
position, and any of them could be the one that degrades.

So test the op on its own. Build a paged K/V cache of a known length, run
`paged_scaled_dot_product_attention_decode` at that position, and compare
against plain softmax attention over the same keys in float64. No model, no
state, nothing else that can drift.
"""
import sys

import torch
import ttnn

BATCH, N_KV, N_Q, HD = 1, 2, 24, 256
BLOCK = 32
MAXLEN = 1024
K_CHUNK = int(sys.argv[1]) if len(sys.argv) > 1 else 128
LENS = (32, 64, 96, 128, 160, 192, 256, 320, 448)

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
torch.manual_seed(0)
rep = ttnn.ReplicateTensorToMesh(mesh)
comp = ttnn.ConcatMeshToTensor(mesh, dim=0)

kern = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
)

n_pages = MAXLEN // BLOCK
page_table = ttnn.from_torch(
    torch.arange(BATCH * n_pages, dtype=torch.int32).reshape(BATCH, n_pages),
    dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep,
)

# Self-consistency instead of an external reference. A first attempt compared
# against hand-written softmax attention and called every length wrong, including
# ones the model handles at 84 % accuracy -- the head-to-KV mapping in the
# reference was wrong, not the op. Comparing the op against *itself* at different
# k_chunk sizes needs no such assumption: for a given cache length the answer must
# not depend on how the online softmax is chunked, so any disagreement is the
# chunking.
KS = (32, 64, 128, 256, 512)


def run(k_chunk, qd, kd, vd, pos):
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        q_chunk_size=32, k_chunk_size=k_chunk, exp_approx_mode=False,
    )
    out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        qd, kd, vd, page_table_tensor=page_table, is_causal=True,
        cur_pos_tensor=pos, scale=HD ** -0.5, program_config=pc,
        compute_kernel_config=kern,
    )
    return ttnn.to_torch(out, mesh_composer=comp)[:1].double()


print(f"RESULT n_q={N_Q} n_kv={N_KV} head_dim={HD}; each length compared across "
      f"k_chunk {KS}", flush=True)
for L in LENS:
    kt = torch.randn(BATCH * n_pages, N_KV, BLOCK, HD) * 0.1
    vt = torch.randn(BATCH * n_pages, N_KV, BLOCK, HD) * 0.1
    flat_k = kt.reshape(n_pages * BLOCK, N_KV, HD)
    flat_v = vt.reshape(n_pages * BLOCK, N_KV, HD)
    flat_k[L:] = 0.0
    flat_v[L:] = 0.0
    qt = torch.randn(1, BATCH, N_Q, HD) * 0.1

    kd = ttnn.from_torch(kt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                         device=mesh, mesh_mapper=rep)
    vd = ttnn.from_torch(vt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                         device=mesh, mesh_mapper=rep)
    qd = ttnn.from_torch(qt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                         device=mesh, mesh_mapper=rep)
    pos = ttnn.from_torch(torch.tensor([L - 1], dtype=torch.int32), dtype=ttnn.int32,
                          layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)

    # An absolute reference, with the GQA mapping *identified* rather than
    # assumed: at L=32 everything fits one k-chunk and the op is known good (the
    # model scores 84 % over those positions), so whichever mapping matches there
    # is the right one, and it is then trusted at longer lengths.
    def reference(mapping):
        kk = flat_k.to(torch.bfloat16).double()[:L]
        vv = flat_v.to(torch.bfloat16).double()[:L]
        qq = qt.to(torch.bfloat16).double().reshape(BATCH, N_Q, HD)
        out = torch.zeros(BATCH, N_Q, HD, dtype=torch.float64)
        for h in range(N_Q):
            kv = (h // (N_Q // N_KV)) if mapping == "block" else (h % N_KV)
            sc = (qq[0, h] @ kk[:, kv, :].T) * (HD ** -0.5)
            out[0, h] = torch.softmax(sc, dim=-1) @ vv[:, kv, :]
        return out

    outs = {}
    for k in KS:
        try:
            outs[k] = run(k, qd, kd, vd, pos)
        except Exception as exc:                              # noqa: BLE001
            outs[k] = None
    base_k = max(k for k in KS if outs[k] is not None)
    base = outs[base_k]
    parts = []
    for k in KS:
        if outs[k] is None:
            parts.append(f"k={k}:n/a")
            continue
        d = (outs[k] - base).abs()
        scale = base.abs().max().item() or 1.0
        parts.append(f"k={k}:{d.max().item() / scale:.2e}")
    refs = {mp: reference(mp) for mp in ("block", "interleave")}
    best = min(refs, key=lambda mp: (outs[128] - refs[mp]).abs().max().item()
               if outs.get(128) is not None else 1e9)
    rparts = []
    for k in KS:
        if outs[k] is None:
            continue
        d = (outs[k] - refs[best]).abs().max().item()
        sc = refs[best].abs().max().item() or 1.0
        rparts.append(f"k={k}:{d / sc:.2e}")
    print(f"RESULT L={L:4d} vs k={base_k}   " + "  ".join(parts), flush=True)
    print(f"RESULT L={L:4d} vs exact ({best})   " + "  ".join(rparts), flush=True)
    for t in (kd, vd, qd):
        ttnn.deallocate(t)

ttnn.close_mesh_device(mesh)
