"""Can paged SDPA read a *short* page table? The whole long-context plan rests on it.

    uv run python scripts/stage9_compact_page_table.py

At the model's maximum context the decode attention reads the entire cache every
layer: 262144 positions x 2 kv heads x 256 head_dim x 2 bytes x 2 (K and V) is
537 MB a layer, 6.4 GB a token over the twelve QSA layers -- 16.6 ms at 388 GB/s
before a single weight is touched. And it reads it *densely*, because
`indexer_max_seq` is 65536: `ttnn.scatter` takes uint16 indices, so the selection
cannot address a longer mask row and `use_indexer` turns itself off. Past 64 k
tokens the model therefore runs plain causal attention, which is both slower and
not the model.

Both problems have one fix. The selection already names 512 blocks of 4 tokens;
those live in at most 512 pages of the 32-token paged cache. Hand SDPA a page
table listing *only those pages* and it reads 17408 positions instead of 262144 --
a constant 1.10 ms whichever context length it is -- and the mask is 17408 wide,
which uint16 addresses comfortably.

Three properties have to hold, and none of them is documented:

 1. a page table with fewer entries than the cache has pages is accepted, with
    `cur_pos` bounded by the compact length rather than the real one;
 2. repeated entries are allowed -- two selected 4-token blocks can share a
    32-token page, and each slot then enables its own four tokens;
 3. the answer matches attention computed directly over the selected positions.

This checks all three against a torch reference before any of it is wired in.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import ttnn

KV_BLOCK = 32
RATIO = 4                  # indexer_compress_ratio
N_KV = 2
N_Q = 8                    # smaller than the model's 24; the question is the API
HD = 128
MAX_SEQ = 4096             # small, so a dense reference is cheap
SLOTS = 16                 # compact slots -> 512 mask columns


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "dev"))
    from _device_model import open_model                              # noqa: E402

    mesh, cfg, m = open_model(max_seq_len=512)
    rep = ttnn.ReplicateTensorToMesh(mesh)
    torch.manual_seed(0)
    try:
        n_pages = MAX_SEQ // KV_BLOCK
        pos = 3000                                   # current position
        # --- a cache with known contents ---------------------------------
        k_host = torch.randn(1, N_KV, MAX_SEQ, HD) * 0.3
        v_host = torch.randn(1, N_KV, MAX_SEQ, HD) * 0.3
        q_host = torch.randn(1, 1, N_Q, HD) * 0.3

        def paged(t):
            # [1, n_kv, T, hd] -> [T/32, n_kv, 32, hd]
            return t.reshape(1, N_KV, n_pages, KV_BLOCK, HD).permute(0, 2, 1, 3, 4)\
                    .reshape(n_pages, N_KV, KV_BLOCK, HD).contiguous()

        keys = ttnn.from_torch(paged(k_host), dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        values = ttnn.from_torch(paged(v_host), dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        q = ttnn.from_torch(q_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=mesh, mesh_mapper=rep)

        # --- a selection: some blocks share a page, deliberately -----------
        # blocks are 4 tokens; pages are 32, so blocks 8..15 all live in page 1.
        blocks = [0, 1, 8, 9, 10, 40, 41, 200, 201, 202, 300, 700, 701, 702,
                  pos // RATIO - 1, pos // RATIO]
        assert len(blocks) == SLOTS
        pages = [b // (KV_BLOCK // RATIO) for b in blocks]
        dup = len(pages) - len(set(pages))
        print(f"RESULT {SLOTS} slots over {len(set(pages))} distinct pages "
              f"({dup} repeated entries)", flush=True)

        table = ttnn.from_torch(
            torch.tensor(pages, dtype=torch.int32).reshape(1, SLOTS),
            dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
            mesh_mapper=rep)

        # --- the mask: within slot j, only that block's own four tokens ----
        width = SLOTS * KV_BLOCK
        mask_host = torch.full((1, N_Q, width), -1e9)
        want = []
        for j, b in enumerate(blocks):
            intra = b % (KV_BLOCK // RATIO)
            for t in range(RATIO):
                tok = b * RATIO + t
                if tok > pos:
                    continue
                mask_host[0, :, j * KV_BLOCK + intra * RATIO + t] = 0.0
                want.append(tok)
        print(f"RESULT {len(want)} distinct positions selected, "
              f"mask {width} wide against a {MAX_SEQ} cache", flush=True)
        mask = ttnn.from_torch(mask_host.reshape(1, 1, N_Q, width), dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        cur = ttnn.from_torch(torch.tensor([width - 1], dtype=torch.int32),
                              dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                              device=mesh, mesh_mapper=rep)

        try:
            out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q, keys, values, page_table_tensor=table, is_causal=False,
                attn_mask=mask, cur_pos_tensor=cur, scale=HD ** -0.5)
        except Exception as exc:                                      # noqa: BLE001
            print(f"RESULT compact page table REJECTED: {type(exc).__name__}: "
                  f"{(str(exc) or repr(exc)).splitlines()[0][:220]}", flush=True)
            return
        got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1]
        print(f"RESULT accepted; output {tuple(got.shape)}", flush=True)

        # --- torch reference over exactly the selected positions ------------
        sel = sorted(set(want))
        kq = k_host[0, :, sel, :].to(torch.float64)          # [n_kv, S, hd]
        vq = v_host[0, :, sel, :].to(torch.float64)
        qq = q_host[0, 0].to(torch.float64)                  # [n_q, hd]
        per = N_Q // N_KV
        ref = torch.zeros(N_Q, HD, dtype=torch.float64)
        for h in range(N_Q):
            logits = (kq[h // per] @ qq[h]) * (HD ** -0.5)
            w = torch.softmax(logits, dim=-1)
            ref[h] = w @ vq[h // per]
        mine = got.reshape(-1, HD)[:N_Q].to(torch.float64)
        err = (mine - ref).abs().max().item()
        scale = ref.abs().max().item()
        print(f"RESULT vs torch over the selected positions: max abs {err:.3e}, "
              f"rel {err / max(scale, 1e-30):.3e}", flush=True)
        print("RESULT " + ("MATCHES -- the compact page table is usable"
                           if err / max(scale, 1e-30) < 3e-2 else
                           "MISMATCH -- the compact page table does not mean what it looks like"),
              flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
