"""Can the QSA attention chunk run on a paged cache? The prerequisite for 5.2.

    uv run python scripts/dev/paged_attention_probe.py

`_attention_chunk` cannot be captured because four of its inputs are host-built:
the rope table, `fill_cache`'s integer `update_idx`, the causal mask, and a
`kv_len` that grows every chunk. `ttnn.experimental.paged_fill_cache` takes the
page table as a *device tensor* and
`ttnn.transformer.chunked_scaled_dot_product_attention` takes
`chunk_start_idx_tensor`, whose own docs introduce it for trace capture -- and
being causal internally it needs no mask and no growing slice, so it removes
three of the four at once.

The cost is a paged K/V cache where decode currently uses a flat one, which is a
refactor of a working path. So this asks the question that decides whether that
refactor is worth starting, at this model's real shapes and against a plain
torch reference: does the paged pair reproduce dense causal attention?
"""
import torch
import ttnn

B, NQ, NKV, DH = 1, 24, 2, 256
BLOCK, T, CHUNK = 32, 512, 128
BLOCKS = T // BLOCK

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
torch.manual_seed(0)


def dn(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, device=mesh)


def hn(t, keep=B):
    """Inputs are replicated, so concatenating and keeping the first shard reads
    one device's copy. `keep` is that copy's extent along dim 0 -- which is the
    block count for a paged cache, not the batch."""
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:keep].float()


def probe(tag, fn):
    try:
        r = fn()
        print(f"PROBE {tag}: OK", flush=True)
        return r
    except Exception as e:
        print(f"PROBE {tag}: FAILED {' '.join(str(e).split())[:200]}", flush=True)
        return None


pc = probe("SDPAProgramConfig", lambda: ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=mesh.compute_with_storage_grid_size(),
    q_chunk_size=CHUNK, k_chunk_size=CHUNK, exp_approx_mode=False,
))

# a paged cache and its page table: one sequence, blocks laid out in order
keys = ttnn.zeros((BLOCKS, NKV, BLOCK, DH), dtype=ttnn.bfloat16,
                  layout=ttnn.TILE_LAYOUT, device=mesh)
values = ttnn.zeros((BLOCKS, NKV, BLOCK, DH), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=mesh)
page_table = dn(torch.arange(BLOCKS, dtype=torch.int32).reshape(B, BLOCKS),
                ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

host_k = torch.randn(B, NKV, T, DH).bfloat16()
host_v = torch.randn(B, NKV, T, DH).bfloat16()
host_q = torch.randn(B, NQ, T, DH).bfloat16()

# fill the cache one CHUNK at a time, exactly as prefill would
for start in range(0, T, CHUNK):
    kk = dn(host_k[:, :, start : start + CHUNK])
    vv = dn(host_v[:, :, start : start + CHUNK])
    ok = probe(f"paged_fill_cache @{start}", lambda kk=kk, vv=vv, s=start: (
        ttnn.experimental.paged_fill_cache(keys, kk, page_table, batch_idx=0,
                                           cache_position_modulo=None) if False else
        ttnn.experimental.paged_fill_cache(keys, kk, page_table, batch_idx=0)
    ))
    if ok is None:
        break

print("\nPROBE --- filling with a per-chunk page table slice instead ---", flush=True)
# `paged_fill_cache` writes the whole input at the *start* of the sequence, so a
# later chunk needs a page table whose first entries are that chunk's blocks.
for start in range(0, T, CHUNK):
    blk = start // BLOCK
    pt = dn(torch.arange(blk, BLOCKS, dtype=torch.int32).reshape(B, -1),
            ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    for cache, src in ((keys, host_k), (values, host_v)):
        ttnn.experimental.paged_fill_cache(
            cache, dn(src[:, :, start : start + CHUNK]), pt, batch_idx=0
        )
print("PROBE filled all chunks", flush=True)

# read the cache back and check it against the host tensors
got_k = hn(keys, BLOCKS).permute(1, 0, 2, 3).reshape(NKV, T, DH)
err = float((got_k - host_k[0].float()).abs().max())
print(f"PROBE cache contents match: {err == 0.0} (max abs {err:g})", flush=True)

# now attention over one chunk, against a torch reference
for start in (0, 128, 256):
    q = dn(host_q[:, :, start : start + CHUNK])
    cs = dn(torch.tensor([start], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    out = probe(f"chunked_sdpa @{start} (tensor start)", lambda q=q, cs=cs: (
        ttnn.transformer.chunked_scaled_dot_product_attention(
            q, keys, values, page_table, chunk_start_idx_tensor=cs,
            scale=DH ** -0.5, program_config=pc,
        )
    ))
    if out is None:
        continue
    got = hn(out)[0]                                   # [NQ, CHUNK, DH]
    kk = host_k[0].float().repeat_interleave(NQ // NKV, dim=0)   # [NQ, T, DH]
    vv = host_v[0].float().repeat_interleave(NQ // NKV, dim=0)
    qq = host_q[0, :, start : start + CHUNK].float()
    total = start + CHUNK
    scores = qq @ kk[:, :total].transpose(-1, -2) * DH ** -0.5
    qpos = torch.arange(start, total).unsqueeze(-1)
    kpos = torch.arange(total).unsqueeze(0)
    scores = scores.masked_fill(kpos > qpos, float("-inf"))
    want = torch.softmax(scores, dim=-1) @ vv[:, :total]
    rel = float((got - want).abs().max() / want.abs().max().clamp(min=1e-9)) * 100
    print(f"PROBE   vs float32 reference: {rel:.3f}%", flush=True)

    # The float32 reference is the wrong yardstick for a bf16 kernel: the error
    # grows with the key count because the accumulation does. The question that
    # matters is whether the paged pair agrees with the *dense device path*
    # `_attention_chunk` runs today -- same dtype, same hardware, tile-padded
    # additive mask and all.
    kv_len = min(-(-total // 32) * 32, T)
    dense_k = dn(host_k[:, :, :kv_len].repeat_interleave(NQ // NKV, dim=1))
    dense_v = dn(host_v[:, :, :kv_len].repeat_interleave(NQ // NKV, dim=1))
    kpos_p = torch.arange(kv_len).unsqueeze(0)
    mask = torch.where(kpos_p <= qpos, 0.0, float("-inf")).reshape(1, 1, CHUNK, kv_len)
    dense = ttnn.transformer.scaled_dot_product_attention(
        q, dense_k, dense_v, attn_mask=dn(mask), is_causal=False, scale=DH ** -0.5,
    )
    dg = hn(dense)[0]
    rel_d = float((got - dg).abs().max() / dg.abs().max().clamp(min=1e-9)) * 100
    rel_dense_ref = float((dg - want).abs().max() / want.abs().max().clamp(min=1e-9)) * 100
    print(f"PROBE   dense device path vs float32: {rel_dense_ref:.3f}%", flush=True)
    print(f"PROBE   paged vs dense device path:   {rel_d:.3f}% "
          f"{'OK -- same answer, different plumbing' if rel_d < 5 else '<-- DIFFERS'}",
          flush=True)

# --------------------------------------------------------------------------
# The decode *write*. `paged_update_cache` already backs the flat path; in paged
# mode it takes `page_table=` (not `page_table_tensor=`, which it rejects) and
# keeps the same precondition -- the input must be L1 height-sharded, one shard
# per core, tile-high, shard width == head_dim, counted from the padded height.
# `TTModel._l1_height_sharded` already builds exactly that.
print("\nPROBE --- decode writes into the paged cache ---", flush=True)


def l1_height_sharded(t, width):
    shape = list(t.shape)
    n = 1
    for d in shape[:-2]:
        n *= d
    n *= (shape[-2] + 31) // 32
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n - 1, 0))})
    spec = ttnn.ShardSpec(grid, [32, width], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.to_memory_config(t, ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec))


WPOS = 300
wrow = torch.randn(1, B, NKV, DH).bfloat16()
wpos = dn(torch.tensor([WPOS] * B, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
probe("paged_update_cache(page_table=)", lambda: ttnn.experimental.paged_update_cache(
    keys, l1_height_sharded(dn(wrow), DH), update_idxs_tensor=wpos, page_table=page_table,
))
back = hn(keys, BLOCKS).permute(1, 0, 2, 3).reshape(NKV, T, DH)
werr = float((back[:, WPOS] - wrow[0, 0].float()).abs().max())
print(f"PROBE   wrote position {WPOS} exactly: {werr == 0.0} (max abs {werr:g})", flush=True)
host_k[0, :, WPOS] = wrow[0, 0]          # keep the reference in step

# --------------------------------------------------------------------------
# The decode half. A paged cache is only worth anything if the *decode* path can
# read it too -- otherwise prefill and decode need two layouts and there is no
# room for both (K/V is 6.4 GB a slot at full context). Both paged variants
# exist, so check them at decode shapes against the flat ops in use today.
print("\nPROBE --- decode on the same paged cache ---", flush=True)
POS = 300
q_dec = torch.randn(1, 1, NQ, DH).bfloat16()          # [1, b, nh, dh]
qd = dn(q_dec)
cur = dn(torch.tensor([POS], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

paged_out = probe("paged_sdpa_decode", lambda: (
    ttnn.transformer.paged_scaled_dot_product_attention_decode(
        qd, keys, values, page_table_tensor=page_table, cur_pos_tensor=cur,
        scale=DH ** -0.5,
    )
))
if paged_out is not None:
    got = hn(paged_out)[0, 0]                          # [NQ, DH]
    kk = host_k[0].float().repeat_interleave(NQ // NKV, dim=0)
    vv = host_v[0].float().repeat_interleave(NQ // NKV, dim=0)
    qq = q_dec[0, 0].float()                           # [NQ, DH]
    scores = (qq.unsqueeze(1) @ kk[:, : POS + 1].transpose(-1, -2)).squeeze(1) * DH ** -0.5
    want = (torch.softmax(scores, dim=-1).unsqueeze(1) @ vv[:, : POS + 1]).squeeze(1)
    rel = float((got - want).abs().max() / want.abs().max().clamp(min=1e-9)) * 100
    print(f"PROBE   paged decode vs float32 reference: {rel:.3f}% "
          f"{'OK' if rel < 8 else '<-- WRONG'}", flush=True)

    # and against the flat decode op the model runs today
    flat_k = dn(host_k[:, :, :T])
    flat_v = dn(host_v[:, :, :T])
    flat_out = probe("flat sdpa_decode (today's path)", lambda: (
        ttnn.transformer.scaled_dot_product_attention_decode(
            qd, flat_k, flat_v, cur_pos_tensor=cur, scale=DH ** -0.5,
        )
    ))
    if flat_out is not None:
        fg = hn(flat_out)[0, 0]
        rel_f = float((got - fg).abs().max() / fg.abs().max().clamp(min=1e-9)) * 100
        print(f"PROBE   paged decode vs flat decode: {rel_f:.3f}% "
              f"{'OK -- one cache can serve both paths' if rel_f < 8 else '<-- DIFFERS'}",
              flush=True)

ttnn.close_mesh_device(mesh)
