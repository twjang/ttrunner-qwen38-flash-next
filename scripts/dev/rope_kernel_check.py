"""Does the fused rope give the same answer as the eleven ops it replaces?

    uv run python scripts/dev/rope_kernel_check.py

The rotation is a swap of two whole tiles -- rope_dim is 64 and half is 32 --
so every output tile is either a fused pair of multiplies or a copy. That makes
the kernel simple and makes a mistake in it easy to miss: the un-rotated
channels pass straight through, so three quarters of the output can be right
while the rotated quarter is not.

Checked per tile column, so the failure says *which* one.
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
ROT = cfg.rope_dim
HD = cfg.head_dim
print(f"RESULT head_dim {HD}, rope_dim {ROT}, half {ROT // 2}", flush=True)

for heads in (2, 24):
    x_h = torch.randn(1, 1, heads, HD) * 0.5
    cos_h = torch.cos(torch.randn(1, 1, 1, ROT))
    sin_h = torch.sin(torch.randn(1, 1, 1, ROT))

    x = ttnn.from_torch(x_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=mesh, mesh_mapper=rep)
    cos = ttnn.from_torch(cos_h, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                          device=mesh, mesh_mapper=rep)
    sin = ttnn.from_torch(sin_h, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                          device=mesh, mesh_mapper=rep)

    def full(t, dtype):
        return ttnn.from_torch(t.expand(1, 1, 32, ROT).contiguous(), dtype=dtype,
                               layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)

    ops._NO_FUSED_ROPE = True
    ref = ttnn.to_torch(m._apply_rope_dev(x, cos, sin), mesh_composer=comp)[:1]
    ops._NO_FUSED_ROPE = False

    # float64 truth on the operands the device holds
    x64 = ttnn.to_torch(x, mesh_composer=comp)[:1].to(torch.float64)
    c64 = cos_h.to(torch.float64)
    s64 = sin_h.to(torch.float64)
    half = ROT // 2
    xr = x64[..., :ROT]
    rot = torch.cat([-xr[..., half:], xr[..., :half]], dim=-1)
    truth = torch.cat([xr * c64 + rot * s64, x64[..., ROT:]], dim=-1)
    scale = truth.abs().max().item()

    for dt_name, dt in (("float32", ttnn.float32), ("bfloat16", ttnn.bfloat16)):
        got_t = ops.fused_rope(x, full(cos_h, dt), full(sin_h, dt), ROT)
        if got_t is None:
            print(f"RESULT heads {heads:2d} {dt_name}: kernel declined", flush=True)
            continue
        ttnn.synchronize_device(mesh)
        got = ttnn.to_torch(got_t, mesh_composer=comp)[:1]
        d_ops = (got.to(torch.float64) - ref.to(torch.float64)).abs().max().item() / scale
        d_t = (got.to(torch.float64) - truth).abs().max().item() / scale
        r_t = (ref.to(torch.float64) - truth).abs().max().item() / scale
        # per tile column, so a failure says which one
        bad = []
        for c in range(HD // 32):
            sl = slice(c * 32, (c + 1) * 32)
            e = (got[..., sl].to(torch.float64) - truth[..., sl]).abs().max().item() / scale
            if e > max(r_t * 2, 1e-6):
                bad.append(f"{c}:{e:.1e}")
        print(f"RESULT heads {heads:2d} {dt_name:8s}: vs ops {d_ops:.2e}; "
              f"vs float64 kernel {d_t:.2e} ops {r_t:.2e}; bad tiles "
              f"[{', '.join(bad) if bad else 'none'}]", flush=True)

ttnn.close_mesh_device(mesh)
