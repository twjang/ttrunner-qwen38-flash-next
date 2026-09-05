"""Shard the experts on the intermediate axis instead of the expert axis.

    uv run python scripts/stage8_expert_shard_axis.py

The wide path fixes `k_sel` at 10 because the top-10 experts are split across
four devices by expert id and *all ten could land on one*. The expected count is
2.5, so three quarters of every gather and both matmuls is padding that carries
zero weight -- 22.2 ms a token of the 33.3 the MoE costs (gather 13.19,
gate|up matmul 3.45, down matmul 5.53).

Sharding the other way removes the worst case rather than paying for it. Give
every device all 512 experts but only its quarter of the intermediate width:

    today   gate|up  [128 experts, 2560, 1280]  gather 10 x 2560 x 1280 = 16.4 MB
    instead gate|up  [512 experts, 2560,  320]  gather 10 x 2560 x  320 =  4.1 MB

Same bytes resident (419 M elements either way), same arithmetic, and the
gathered part is now *all* useful: every device runs all ten selected experts
over a quarter of their columns. The combine still works -- the down projection
is already sharded on its contraction dim and the caller already all-reduces;
this only changes what the partial sum is over, experts becoming columns.

No approximation anywhere, so the only question is whether it is faster. This
measures the gather and both matmuls at both shapes before anything is
re-converted, because re-sharding the weight cache is the expensive part.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Today's per-device worst case, and the global one the other layout needs.
# `kept_expert_census.py` measures the tie admission over 2256 real routing
# decisions: 10 experts kept 77 % of the time, 11 18 %, up to a maximum of 14.
# So a *global* k_sel of 16 drops nothing on that sample and happens to be the
# gather kernel's own limit -- the selection is read from one tile face.
K_SEL_EXPERT_AXIS = 10
K_SEL_GLOBAL = 16
HIDDEN = 2560
INTER = 640                # expert_intermediate
N_DEV = 4
E_TOTAL = 512


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "dev"))
    from _device_model import open_model                              # noqa: E402
    import ttrunner_qwen38_flash_next.tt.moe as moe                   # noqa: E402

    mesh, cfg, m = open_model(max_seq_len=512)
    rep = ttnn.ReplicateTensorToMesh(mesh)
    torch.manual_seed(0)
    try:
        e_local = E_TOTAL // N_DEV

        # (label, experts held, gate|up columns, down rows)
        LAYOUTS = [
            ("expert-axis  (today)", e_local, 2 * INTER, INTER, K_SEL_EXPERT_AXIS),
            ("intermediate-axis", E_TOTAL, 2 * INTER // N_DEV, INTER // N_DEV,
             K_SEL_GLOBAL),
        ]
        def make_idx(k):
            return ttnn.from_torch(
                torch.arange(k, dtype=torch.int32).reshape(1, 1, 1, k) * 7,
                dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        x = ttnn.from_torch(torch.randn(1, 1, 1, HIDDEN) * 0.05, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)

        def timed(fn, tag, reps=30, iters=8):
            for _ in range(2):
                fn()
            ttnn.synchronize_device(mesh)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            for _ in range(reps):
                fn()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            best = float("inf")
            for _ in range(iters):
                t0 = time.perf_counter()
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                best = min(best, time.perf_counter() - t0)
            ttnn.release_trace(mesh, tid)
            us = best / reps * 1e6
            print(f"RESULT   {tag:32s} {us:8.2f} us  ({48 * us / 1000:6.2f} ms a token)",
                  flush=True)
            return us

        totals = {}
        for label, n_e, gu_cols, dn_rows, K_SEL in LAYOUTS:
            idx = make_idx(K_SEL)
            print(f"RESULT {label}: {n_e} experts, k_sel {K_SEL}, "
                  f"gate|up [.., {HIDDEN}, {gu_cols}], down [.., {dn_rows}, {HIDDEN}]",
                  flush=True)
            gate_w = ttnn.from_torch(
                torch.randn(1, n_e, HIDDEN, gu_cols) * 0.02, dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
            down_w = ttnn.from_torch(
                torch.randn(1, n_e, dn_rows, HIDDEN) * 0.02, dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
            print(f"RESULT   resident {(gate_w.volume() + down_w.volume()) / 2 / 2**20:.1f} "
                  f"MiB a device, gathered "
                  f"{K_SEL * HIDDEN * gu_cols / 2 / 2**20 + K_SEL * dn_rows * HIDDEN / 2 / 2**20:.1f} MiB",
                  flush=True)

            gu_shape = (1, 1, HIDDEN, K_SEL * gu_cols)
            dn_shape = (1, 1, K_SEL * gu_cols // 2, HIDDEN)

            def gather_both(gate_w=gate_w, down_w=down_w, gu_shape=gu_shape,
                            dn_shape=dn_shape):
                moe._gather(gate_w, idx, K_SEL, 2, gu_shape, 1)
                moe._gather(down_w, idx, K_SEL, 0, dn_shape, 1)

            gather_both()
            gu = moe._buf(("g", 2, gu_shape, str(gate_w.dtype)), gu_shape,
                          gate_w.dtype, mesh)
            dw = moe._buf(("g", 0, dn_shape, str(down_w.dtype)), dn_shape,
                          down_w.dtype, mesh)
            hid = ttnn.from_torch(
                torch.zeros(1, 1, 1, K_SEL * gu_cols // 2), dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)

            g = timed(gather_both, "both gathers")
            l1 = timed(lambda gu=gu: ttnn.linear(x, gu, compute_kernel_config=moe.HIFI4),
                       "gate|up matmul")
            l2 = timed(lambda dw=dw, hid=hid: ttnn.linear(
                hid, dw, compute_kernel_config=moe.HIFI4), "down matmul")
            totals[label] = g + l1 + l2
            print(f"RESULT   total {48 * (g + l1 + l2) / 1000:6.2f} ms a token", flush=True)
            ttnn.deallocate(gate_w)
            ttnn.deallocate(down_w)
            moe._GATHER_BUF.clear()

        a, b = totals[LAYOUTS[0][0]], totals[LAYOUTS[1][0]]
        print("RESULT ---", flush=True)
        print(f"RESULT {48 * a / 1000:.2f} -> {48 * b / 1000:.2f} ms a token "
              f"({a / b:.2f}x, {48 * (a - b) / 1000:+.2f} ms)", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
