"""Does the fused reinject give the same answer as the four ops it replaces?

    uv run python scripts/dev/reinject_kernel_check.py

`reinject` runs 96 times a token -- twice a layer -- and as four ttnn ops it
materialises a [.., hc, M, hidden] intermediate twice. At M=1 that stack is
padded from one row to thirty-two, so each op moves ~655 KB to carry 10 KB, and
one of the four is a permute of the whole thing.

The kernel does it in one pass, and the permute simply does not exist: a writer
chooses its destination tile.

Checked at the decode shape and at a multi-row shape, because the per-row
injection scalar is the part most likely to be wrong -- the reader synthesises a
tile whose row m is filled with inject[h, m], and a bug there would be invisible
at M=1 where every row holds the same value.
"""
import os
import sys

import torch
import ttnn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _device_model import open_model                                 # noqa: E402

mesh, cfg, m = open_model(max_seq_len=512)
import ttrunner_qwen38_flash_next.tt.ops as ops                      # noqa: E402

HC, HIDDEN = cfg.hc_count, cfg.hidden_size
rep = ttnn.ReplicateTensorToMesh(mesh)
torch.manual_seed(0)
print(f"RESULT hc {HC}, hidden {HIDDEN}", flush=True)

ok = True
for M in (1, 8, 32):
    hyper_h = torch.randn(1, 1, M, HC * HIDDEN) * 0.5
    branch_h = torch.randn(1, 1, M, HIDDEN) * 0.5
    inject_h = torch.randn(1, HC, M, 1) * 0.5

    def dev(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=mesh, mesh_mapper=rep)

    hyper, branch, inject = dev(hyper_h), dev(branch_h), dev(inject_h)

    ops._NO_FUSED_REINJECT = True
    ref = ttnn.to_torch(ops.reinject(hyper, branch, inject, HC),
                        mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1]
    ops._NO_FUSED_REINJECT = False
    got_t = ops.reinject(hyper, branch, inject, HC)
    ttnn.synchronize_device(mesh)
    got = ttnn.to_torch(got_t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1]

    # and against float64, since "differs from the ops" is not "wrong"
    truth = (hyper_h.to(torch.float64)
             + (inject_h.to(torch.float64) * branch_h.to(torch.float64))
             .permute(0, 2, 1, 3).reshape(1, 1, M, HC * HIDDEN))
    scale = truth.abs().max().item()

    def rel(t):
        return (t.to(torch.float64) - truth).abs().max().item() / max(scale, 1e-30)

    d = (got.to(torch.float64) - ref.to(torch.float64)).abs().max().item() / max(scale, 1e-30)
    print(f"RESULT M={M:2d}: vs the four ops {d:.3e}; vs float64 kernel {rel(got):.3e}, "
          f"ops {rel(ref):.3e}", flush=True)
    if rel(got) > max(rel(ref) * 1.5, 1e-6):
        ok = False
        print(f"RESULT   !! the kernel is further from the truth than the ops", flush=True)

# --- the raw form: (gate stream, first column) rather than a built tensor ----
#
# `gated_residual_mix` now hands the un-sliced stream over and the kernel reads
# column `base + h` of it, applying the 2*sigmoid itself. Four ttnn ops a call
# disappear, so this has to reproduce all four.
SPAN = HIDDEN
for M in (1, 8, 32):
    hyper_h = torch.randn(1, 1, M, HC * HIDDEN) * 0.5
    branch_h = torch.randn(1, 1, M, HIDDEN) * 0.5
    whole_h = torch.randn(1, 1, M, SPAN + 32) * 0.5

    def dev(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=mesh, mesh_mapper=rep)

    hyper, branch, whole = dev(hyper_h), dev(branch_h), dev(whole_h)

    ops._NO_FUSED_REINJECT = True
    ref = ttnn.to_torch(ops.reinject(hyper, branch, (whole, SPAN), HC),
                        mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1]
    ops._NO_FUSED_REINJECT = False
    got = ttnn.to_torch(ops.reinject(hyper, branch, (whole, SPAN), HC),
                        mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1]
    ttnn.synchronize_device(mesh)

    inj64 = 2.0 * torch.sigmoid(whole_h[..., SPAN:SPAN + HC].to(torch.float64))
    truth = (hyper_h.to(torch.float64)
             + (inj64.permute(0, 1, 3, 2).unsqueeze(-1)
                * branch_h.to(torch.float64).unsqueeze(2)).squeeze(1)
             .permute(0, 2, 1, 3).reshape(1, 1, M, HC * HIDDEN))
    scale = truth.abs().max().item()

    def rel2(t):
        return (t.to(torch.float64) - truth).abs().max().item() / max(scale, 1e-30)

    d = (got.to(torch.float64) - ref.to(torch.float64)).abs().max().item() / max(scale, 1e-30)
    print(f"RESULT raw M={M:2d}: vs the ops {d:.3e}; vs float64 kernel {rel2(got):.3e}, "
          f"ops {rel2(ref):.3e}", flush=True)
    if rel2(got) > max(rel2(ref) * 1.5, 1e-6):
        ok = False
        print("RESULT   !! the raw path is further from the truth than the ops", flush=True)

print("RESULT " + ("the fused reinject matches" if ok else "MISMATCH"), flush=True)
ttnn.close_mesh_device(mesh)
