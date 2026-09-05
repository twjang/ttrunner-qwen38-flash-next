"""What would `attn_qkv` cost if the v-heads were regrouped by k-head?

    uv run python scripts/dev/qkv_regroup_price.py

`convert.py` gives device d the v-heads [12d, 12d+12), and twelve consecutive
v-heads touch twelve *distinct* k-heads mod 16 -- so each device stores
q 1536 + k 1536 + v 1536 = 4608 columns where grouping by k-head would need
q 512 + k 512 + v 1536 = 2560. The mesh holds 12288 q|k columns for 4096
distinct ones (handoff 45.1).

The regroup is a re-conversion of six tensor families and a kernel guard, so it
is worth an hour of measurement first: this times the two shapes directly, at
M = 1, through the same `fast_linear` the model uses. No weights are loaded and
no model is built -- a mesh, two random tensors, and the program config.

36 layers a token, so the per-call difference times 36 is the whole prize.
"""
import sys
import time

import torch
import ttnn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ttrunner_qwen38_flash_next.tt.ops import HIFI4, fast_linear     # noqa: E402

LAYERS = 36
REPS = 200
SHAPES = [
    (2560, 4608, "attn_qkv today: q|k stored three times over"),
    (2560, 2560, "attn_qkv regrouped: q|k once"),
    (2560, 1536, "attn_gate, for scale"),
    # The same question for the shared expert, whose sharding is committed but
    # not yet reconverted: does a quarter of the bytes buy a quarter of the time,
    # or does the narrower output just empty the grid?
    (2560, 1312, "shexp gate|up replicated"),
    (2560, 352, "shexp gate|up column-sharded"),
    (640, 2560, "shexp down replicated"),
    (160, 2560, "shexp down row-sharded"),
]


def main() -> None:
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    try:
        xs = {}
        def act(k):
            if k not in xs:
                xs[k] = ttnn.from_torch(
                    torch.randn(1, 1, 1, k), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=mesh,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            return xs[k]

        out = {}
        for k, n, label in SHAPES:
            x = act(k)
            w = ttnn.from_torch(
                torch.randn(1, 1, k, n), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
                device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            for _ in range(10):                       # compile + warm
                fast_linear(x, w, compute_kernel_config=HIFI4)
            ttnn.synchronize_device(mesh)
            # A sync per call measures the sync: all three shapes came back at
            # ~129 us and 0.1 GB/s, which is `synchronize_device`, not the
            # matmul. Issue a run of calls and sync once, then divide.
            batches = []
            for _ in range(9):
                ttnn.synchronize_device(mesh)
                t0 = time.perf_counter()
                for _ in range(REPS):
                    fast_linear(x, w, compute_kernel_config=HIFI4)
                ttnn.synchronize_device(mesh)
                batches.append(1e6 * (time.perf_counter() - t0) / REPS)
            batches.sort()
            us = batches[len(batches) // 2]
            mb = k * n * 1.0625 / 1e6
            # `fast_linear` is nearly flat in N here -- 12.53 MB and 0.96 MB
            # cost the same 30-36 us -- so the bound is the K loop, not bytes.
            # `ksgemv` splits that loop across cores, which is the only thing
            # that changes it. Time both.
            print(f"RESULT [{k:5d},{n:5d}] {us:7.2f} us  {mb:6.2f} MB  "
                  f"{mb / us * 1e6 / 1e3:5.0f} GB/s  {label}", flush=True)
            out[(k, n)] = us
        gain = out[(2560, 4608)] - out[(2560, 2560)]
        print(f"RESULT regroup saves {gain:.2f} us a call, "
              f"{gain * LAYERS / 1000:.3f} ms a token over {LAYERS} layers", flush=True)
        print("RESULT note: isolated timings over-price by 2-4x (invariant 89); "
              "this is an upper bound", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
