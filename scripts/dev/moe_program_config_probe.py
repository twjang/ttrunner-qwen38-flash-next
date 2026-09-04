"""Can a different `sparse_program_config` make row-groups past one tile correct?

    uv run python scripts/dev/moe_program_config_probe.py [layer]      (default 0)

`moe.moe_block` reproduces a per-row MoE exactly for 1, 8, 16 and 32 rows and
gets 33 of 64 rows wrong at 64 (`moe_rows_check.py`). Routing is not the cause --
`keep`, `weights` and the expert union all match the host at 64. What changes at
that boundary is `per_core_M`, which is `ceil(m / 32)`: 1 up to 32 rows, 2 at 64,
4 at 128. The baseline config pairs that with `out_subblock_h=1`,
`fuse_batch=False` and `mcast_in0=True`.

So: hold the rows fixed, vary only the program config, and see whether any of
them reproduces the per-row answer at 64 rows. If one does, `_MAX_MOE_CHUNK`
comes off and prefill gets the rest of its speedup; if none does, the cap is
forced by the op and this says so in the place the next reader will look.
"""
import sys

import torch
import ttnn

from _device_model import open_model

from ttrunner_qwen38_flash_next.tt import moe

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 0
ROWS = 64

mesh, cfg, m = open_model(max_seq_len=512)
torch.manual_seed(0)
host_x = (torch.randn(1, 1, ROWS, cfg.hidden_size) * 0.05).bfloat16()
router_w = m.w.blk(LAYER, "ffn_gate_inp.weight")
gate_w = m.w.fused_gate_up(LAYER) if m.fuse_expert_gate_up else m.w.blk(LAYER, "ffn_gate_exps.weight")
up_w = None if m.fuse_expert_gate_up else m.w.blk(LAYER, "ffn_up_exps.weight")
down_w = m.w.blk(LAYER, "ffn_down_exps.weight")
_MM1D = type(moe.sparse_program_config(32, 2560, 640))
base_cfg = moe.sparse_program_config


def variant(name, **over):
    def make(m_, k, n, grid_x=10, grid_y=8):
        n_tiles, k_tiles, m_tiles = -(-n // 32), -(-k // 32), -(-m_ // 32)
        cores = min(grid_x * grid_y, n_tiles)
        per_core_n = (n_tiles + cores - 1) // cores
        in0_block_w = next((b for b in (8, 5, 4, 2, 1) if k_tiles % b == 0), 1)
        kw = dict(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
            in0_block_w=in0_block_w, out_subblock_h=1, out_subblock_w=1,
            per_core_M=m_tiles, per_core_N=per_core_n,
            fuse_batch=False, mcast_in0=True,
        )
        if "out_subblock_h" in over:
            h = over["out_subblock_h"]
            kw["out_subblock_h"] = h if m_tiles % h == 0 else 1
        for key in ("fuse_batch", "mcast_in0"):
            if key in over:
                kw[key] = over[key]
        return _MM1D(**kw)
    return name, make


def run(rows, start=0):
    x = ttnn.from_torch(
        host_x[:, :, start : start + rows].contiguous(), dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=m.replicate,
    )
    out = moe.moe_block(
        x, router_w, gate_w, up_w, down_w, cfg.num_experts_per_tok,
        cfg.num_experts, cfg.hidden_size, cfg.expert_intermediate,
    )
    return ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1].float()


print("RESULT building the per-row reference", flush=True)
ref = torch.cat([run(1, start=i) for i in range(ROWS)], dim=-2)
print(f"RESULT {'variant':<26} {'worst row':>10} {'rows over 1%':>13}", flush=True)
for name, make in [
    ("baseline (subblock_h=1)", base_cfg),
    *[variant(n, **o) for n, o in (
        ("out_subblock_h=2", {"out_subblock_h": 2}),
        ("out_subblock_h=4", {"out_subblock_h": 4}),
        ("fuse_batch=True", {"fuse_batch": True}),
        ("mcast_in0=False", {"mcast_in0": False}),
    )],
]:
    moe.sparse_program_config = make
    try:
        got = run(ROWS)
        scale = ref.abs().max().clamp(min=1e-9)
        per_row = (got[0, 0] - ref[0, 0]).abs().amax(dim=-1) / scale * 100.0
        bad = int((per_row > 1.0).sum())
        print(f"RESULT {name:<26} {float(per_row.max()):9.3f}% {bad:9d}/{ROWS}"
              f"{'   <-- CORRECT' if bad == 0 else ''}", flush=True)
    except Exception as exc:
        print(f"RESULT {name:<26} rejected: {' '.join(str(exc).split())[:90]}", flush=True)
    finally:
        moe.sparse_program_config = base_cfg
ttnn.close_mesh_device(mesh)
