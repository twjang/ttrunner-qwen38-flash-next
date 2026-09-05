"""Do the right-sized elementwise helpers give the same answers as the ops?

    uv run python scripts/dev/small_ew_check.py

They exist because a launch costs by core count and ttnn takes the whole grid
(`dispatch_floor.py`), so a one-tile op is 5.8 us where 2.06 would do. That is
worth ~10 ms of a 49 ms step, and none of it is worth anything if the arithmetic
moves.

Every helper against its ttnn spelling *and* against float64, because differing
from ttnn is not the same as being wrong (invariant 57).
"""
import sys

import torch
import ttnn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _device_model import open_model                                 # noqa: E402

mesh, cfg, m = open_model(max_seq_len=512)
import ttrunner_qwen38_flash_next.tt.ops as ops                      # noqa: E402

rep = ttnn.ReplicateTensorToMesh(mesh)
comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
torch.manual_seed(0)

SHAPES = [(1, 1, 1, 32), (1, 1, 1, 128), (1, 1, 32, 320), (1, 1, 1, 2048)]
ok = True
for shape in SHAPES:
    ah = torch.randn(*shape)
    bh = torch.randn(*shape)
    a = ttnn.from_torch(ah, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=mesh, mesh_mapper=rep)
    b = ttnn.from_torch(bh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=mesh, mesh_mapper=rep)
    a64 = ttnn.to_torch(a, mesh_composer=comp)[:1].to(torch.float64)
    b64 = ttnn.to_torch(b, mesh_composer=comp)[:1].to(torch.float64)

    cases = [
        ("mul", lambda: ops.ew_mul(a, b), lambda: ttnn.multiply(a, b), a64 * b64),
        ("add", lambda: ops.ew_add(a, b), lambda: ttnn.add(a, b), a64 + b64),
        ("sigmoid", lambda: ops.ew_sigmoid(a), lambda: ttnn.sigmoid(a),
         torch.sigmoid(a64)),
        ("silu", lambda: ops.ew_silu(a), lambda: ttnn.silu(a),
         a64 * torch.sigmoid(a64)),
        ("scale x2", lambda: ops.ew_scale(a, 2.0), lambda: ttnn.multiply(a, 2.0),
         a64 * 2.0),
        ("sigmoid x2", lambda: ops.ew_sigmoid_scale(a, 2.0),
         lambda: ttnn.multiply(ttnn.sigmoid(a), 2.0), torch.sigmoid(a64) * 2.0),
    ]
    if shape[-1] >= 64:
        half = (shape[-1] // 64) * 32
        cases.append(("slice", lambda: ops.ew_slice_last(a, half, shape[-1]),
                      lambda: ttnn.slice(a, (0, 0, 0, half),
                                         (shape[0], shape[1], shape[2], shape[-1])),
                      a64[..., half:]))
    for name, mine, theirs, truth in cases:
        g = ttnn.to_torch(mine(), mesh_composer=comp)[:1].to(torch.float64)
        r = ttnn.to_torch(theirs(), mesh_composer=comp)[:1].to(torch.float64)
        sc = max(truth.abs().max().item(), 1e-30)
        e_k = (g - truth).abs().max().item() / sc
        e_t = (r - truth).abs().max().item() / sc
        d = (g - r).abs().max().item() / sc
        bad = e_k > max(e_t * 1.5, 1e-6)
        ok = ok and not bad
        print(f"RESULT {str(shape):16s} {name:11s} vs ops {d:.2e}  float64: "
              f"kernel {e_k:.2e} ops {e_t:.2e}{'   <-- WORSE' if bad else ''}",
              flush=True)
    for t in (a, b):
        ttnn.deallocate(t)

print("RESULT " + ("all helpers match" if ok else "SOME HELPER IS WORSE"), flush=True)
ttnn.close_mesh_device(mesh)
