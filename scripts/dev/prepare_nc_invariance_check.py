"""Is `prepare_device` bit-identical between one chunk and several?

    uv run python scripts/dev/prepare_nc_invariance_check.py

`prepare_device` is 158.9 ms of a 922 ms prefill chunk -- the single largest item
-- and it is already vectorised over the chunk axis: its inputs are
[H, NC, C, D]. Four chunks in one call issue the same ~59 dispatches as one, so
on a dispatch-bound path (invariant 21) that is most of a 10 % win, *if* raising
NC does not change the answer.

That is not obvious. Raising the row count of a dense `ttnn.linear` does change
each row's result once past 128 (`row_count_stability_check.py`), because the
op picks a different blocking. NC is a batch axis rather than a row axis, so it
should be different -- but "should" is exactly the word this project has been
wrong about before, and the whole plan rests on it.

Compares chunk 0's eight prepared tensors between an NC=1 call and an NC=4 call
built from the same data.
"""
import torch
import ttnn

from twtest.tt.deltanet import CHUNK, prepare_device

H, NC, D = 12, 4, CHUNK

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
torch.manual_seed(0)
rep = ttnn.ReplicateTensorToMesh(mesh)


def dev(t):
    return ttnn.from_torch(t.contiguous(), dtype=ttnn.float32,
                           layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)


q = torch.randn(H, NC, CHUNK, D) * 0.1
k = torch.randn(H, NC, CHUNK, D) * 0.1
v = torch.randn(H, NC, CHUNK, D) * 0.1
g = -torch.rand(H, NC, CHUNK, 1) * 0.05          # decays are negative
b = torch.rand(H, NC, CHUNK, 1)

wide = prepare_device(dev(q), dev(k), dev(v), dev(g), dev(b), mesh=mesh)
one = prepare_device(dev(q[:, :1]), dev(k[:, :1]), dev(v[:, :1]),
                     dev(g[:, :1]), dev(b[:, :1]), mesh=mesh)

comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
ok = True
for name in one:
    a = ttnn.to_torch(one[name], mesh_composer=comp)[:H]
    bb = ttnn.to_torch(wide[name], mesh_composer=comp)[:H]
    bb = bb[:, :1]                                # chunk 0 of the wide call
    same = torch.equal(a, bb)
    ok &= same
    d = (a.float() - bb.float()).abs()
    print(f"RESULT {name:12s} {tuple(a.shape)} vs {tuple(bb.shape)}  "
          f"{'IDENTICAL' if same else 'differs'}  max {d.max().item():.3e}", flush=True)

print(f"RESULT all eight identical: {'YES' if ok else 'NO'}", flush=True)
ttnn.close_mesh_device(mesh)
