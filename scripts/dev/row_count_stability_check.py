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

K, N = 2560, 2048
MS = (32, 64, 128, 256, 384, 512)

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
torch.manual_seed(0)
rep = ttnn.ReplicateTensorToMesh(mesh)
tile = dict(layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)

big = torch.randn(1, 1, max(MS), K) * 0.1
w = ttnn.from_torch(torch.randn(1, 1, K, N) * 0.05, dtype=ttnn.bfloat16, **tile)
comp = ttnn.ConcatMeshToTensor(mesh, dim=0)

ref = None
for m in MS:
    x = ttnn.from_torch(big[:, :, :m, :], dtype=ttnn.bfloat16, **tile)
    out = ttnn.to_torch(ttnn.linear(x, w, compute_kernel_config=HIFI4),
                        mesh_composer=comp)[:1]
    head = out[:, :, :32, :]                       # rows every run shares
    if ref is None:
        ref, ref_m = head, m
        print(f"RESULT m={m:4d}  (reference)", flush=True)
    else:
        same = torch.equal(head, ref)
        d = (head - ref).abs()
        print(f"RESULT m={m:4d}  rows 0-31 vs m={ref_m}: "
              f"{'IDENTICAL' if same else 'differs'}  max {d.max().item():.3e}  "
              f"differing {(d > 0).sum().item()}/{d.numel()}", flush=True)
    ttnn.deallocate(x)

# and the pairwise question that actually matters: 128 against 256/384/512
print("RESULT --- the question for the prefill chunk: is 128 == 256 == 512? ---",
      flush=True)
outs = {}
for m in (128, 256, 512):
    x = ttnn.from_torch(big[:, :, :m, :], dtype=ttnn.bfloat16, **tile)
    outs[m] = ttnn.to_torch(ttnn.linear(x, w, compute_kernel_config=HIFI4),
                            mesh_composer=comp)[:1][:, :, :128, :]
    ttnn.deallocate(x)
for m in (256, 512):
    d = (outs[m] - outs[128]).abs()
    print(f"RESULT rows 0-127: m={m} vs m=128  "
          f"{'IDENTICAL' if torch.equal(outs[m], outs[128]) else 'differs'}  "
          f"max {d.max().item():.3e}  differing {(d > 0).sum().item()}/{d.numel()}",
          flush=True)

# Which of the two is right? Neither is "the" answer -- they are two blockings of
# the same sum -- so compare both to the exact product in float64, built from the
# device's own bf16 operands (which convert exactly), per invariant 20. Widening
# the chunk is only allowed if it does not cost accuracy.
xt = big[:, :, :128, :].to(torch.bfloat16).double()
wt = ttnn.to_torch(w, mesh_composer=comp)[:1].double()
exact = xt[0, 0] @ wt[0, 0]
for m in (128, 256, 512):
    err = (outs[m][0, 0].double() - exact).abs()
    print(f"RESULT vs float64 exact, rows 0-127 at m={m:4d}: "
          f"max {err.max().item():.4e}   mean {err.mean().item():.4e}", flush=True)

ttnn.close_mesh_device(mesh)
