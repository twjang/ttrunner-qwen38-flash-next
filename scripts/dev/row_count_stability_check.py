"""Does a row's `ttnn.linear` result keep changing as the row count grows?

    uv run python scripts/dev/row_count_stability_check.py

Invariant 13 records that a plain `ttnn.linear` returns a given row identically
at m = 1, 8, 16, 32 and differently at 33+ -- one bf16 ulp, the row-tile
boundary. Everything built on that has assumed the answer keeps moving with m,
which is why the prefill chunk has never been widened past 128.

That assumption was never tested. If the result is stable for *all* m above one
tile, then widening the prefill chunk is bit-identical, and it is worth a lot:
prefill is dispatch-bound (invariant 21) and `prepare_device` -- 17 % of a chunk
-- is already vectorised over the chunk count, so 512 tokens in one call issues
the same 59 ops over four times the data instead of 59 ops four times.

Compares the first 128 rows of the same input across row counts, on the shapes
the model actually uses.
"""
import torch
import ttnn

from twtest.tt.ops import HIFI4

# The shapes the model's dense prefill linears actually use, per device.
# hidden 2560, hc_count 4 (so the hyper-connection mixes are 10240 wide), 24 q
# heads x 256 / 4 devices for q|gate, 2 kv heads, DeltaNet 48 v-heads / 4.
SHAPES = [
    ("hc mix down", 10240, 32),
    ("hc mix up", 32, 10240),
    ("hc inject", 10240, 2560),
    ("qsa q|gate", 2560, 3072),
    ("qsa k/v", 2560, 128),
    ("qsa out", 1536, 2560),
    ("deltanet qkv", 2560, 2048),
    ("deltanet out", 1536, 2560),
    ("router", 2560, 512),
    ("shared gate/up", 2560, 160),
]
MS = (128, 256, 512)

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
torch.manual_seed(0)
rep = ttnn.ReplicateTensorToMesh(mesh)
tile = dict(layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
comp = ttnn.ConcatMeshToTensor(mesh, dim=0)

print("RESULT shape                K     N   128vs256   err@128     err@256    verdict",
      flush=True)
worse = better = same = 0
for label, K, N in SHAPES:
    big = torch.randn(1, 1, max(MS), K) * 0.1
    w = ttnn.from_torch(torch.randn(1, 1, K, N) * 0.05, dtype=ttnn.bfloat16, **tile)
    outs = {}
    for m in MS:
        x = ttnn.from_torch(big[:, :, :m, :], dtype=ttnn.bfloat16, **tile)
        outs[m] = ttnn.to_torch(ttnn.linear(x, w, compute_kernel_config=HIFI4),
                                mesh_composer=comp)[:1][:, :, :128, :]
        ttnn.deallocate(x)
    id256 = torch.equal(outs[128], outs[256])
    id512 = torch.equal(outs[128], outs[512])
    identical = id256 and id512
    # exact product in float64 from the device's own operands
    xt = big[:, :, :128, :].to(torch.bfloat16).double()
    wt = ttnn.to_torch(w, mesh_composer=comp)[:1].double()
    exact = xt[0, 0] @ wt[0, 0]
    e128 = (outs[128][0, 0].double() - exact).abs().mean().item()
    e256 = (outs[256][0, 0].double() - exact).abs().mean().item()
    e512 = (outs[512][0, 0].double() - exact).abs().mean().item()
    if identical:
        verdict, same = "identical", same + 1
    elif e256 < e128 * 0.999:
        verdict, better = "256 closer", better + 1
    elif e256 > e128 * 1.001:
        verdict, worse = "256 WORSE", worse + 1
    else:
        verdict, same = "wash", same + 1
    print(f"RESULT {label:16s} {K:5d} {N:5d}  {'same' if identical else 'differs':8s}  "
          f"{e128:.4e}  {e256:.4e}  {verdict}  "
          f"[256 {'=' if id256 else 'x'} 512 {'=' if id512 else 'x'}]  "
          f"err@512 {e512:.4e}", flush=True)
    ttnn.deallocate(w)

print(f"RESULT --- {same} identical-or-wash, {better} better at 256, {worse} worse ---",
      flush=True)
ttnn.close_mesh_device(mesh)
