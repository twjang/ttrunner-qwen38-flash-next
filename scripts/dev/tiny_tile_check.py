"""Can the expert output drop its 32x row padding at M = 1?

    uv run python scripts/dev/tiny_tile_check.py

The routed MoE is 65 % of the traced decode step, and ablation puts most of that
in one tensor rather than in the experts' arithmetic: `expert_ffn`'s down
projection returns [1, E, M, K], and TILE_LAYOUT pads the row axis to 32. At
decode M is 1, so [1, 512, 1, 2560] occupies 84 MB to hold 2.6 MB of answer, it
is written once by the matmul and then read, rewritten and read again by the
combine (`multiply` then `sum` over E: 55.1 ms of a 238 ms step, of which the
permute is 0.5).

`sparse_matmul` takes an `output_tile`. A 1x32 tile would store exactly the rows
that exist. The arithmetic is unchanged, so it should be bit-identical -- and
the downstream `multiply`/`sum` have to accept the tile too, which is the part
that is not obvious.
"""
import time

import torch
import ttnn

from ttrunner_qwen38_flash_next.tt.moe import sparse_program_config, HIFI4

E, M, N, K = 512, 1, 160, 2560
TOPK = 12

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
torch.manual_seed(0)
rep = ttnn.ReplicateTensorToMesh(mesh)
tile = dict(layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)

h = ttnn.from_torch(torch.randn(1, E, M, N) * 0.1, dtype=ttnn.bfloat16, **tile)
dw = ttnn.from_torch(torch.randn(1, E, N, K) * 0.05, dtype=ttnn.bfloat16, **tile)
keep = torch.zeros(1, 1, 1, E)
keep[..., torch.randperm(E)[:TOPK]] = 1.0
sparsity = ttnn.to_layout(ttnn.from_torch(keep, dtype=ttnn.bfloat16, **tile),
                          ttnn.ROW_MAJOR_LAYOUT)
gate = ttnn.from_torch(torch.rand(1, E, M, 1), dtype=ttnn.bfloat16, **tile)
pc = sparse_program_config(M, N, K)
kw = dict(sparsity=sparsity, nnz=None, is_input_a_sparse=True, is_input_b_sparse=True,
          program_config=pc, compute_kernel_config=HIFI4)


def _tile(spec):
    """`output_tile` wants a Tile, not a list -- try both spellings."""
    try:
        return ttnn.Tile(spec)
    except Exception:                                       # noqa: BLE001
        return spec


def down(output_tile=None):
    extra = {"output_tile": _tile(output_tile)} if output_tile else {}
    return ttnn.sparse_matmul(h, dw, **kw, **extra)


def combine(p):
    return ttnn.sum(ttnn.multiply(p, gate), dim=1, keepdim=True)


comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
base = down()
print(f"RESULT default tile: out {tuple(base.shape)}", flush=True)

for cand in ([1, 32], [2, 32], [4, 32], [8, 32], [16, 32]):
    try:
        t = down(cand)
    except Exception as exc:                                # noqa: BLE001
        print(f"RESULT output_tile={cand}: matmul rejected -- "
              f"{type(exc).__name__}: {(str(exc) or repr(exc)).splitlines()[0][:160]}",
              flush=True)
        continue
    try:
        c_t = combine(t)
    except Exception as exc:                                # noqa: BLE001
        print(f"RESULT output_tile={cand}: matmul OK, combine rejected -- "
              f"{type(exc).__name__}: {str(exc).splitlines()[0][:110]}", flush=True)
        continue
    a = ttnn.to_torch(combine(base), mesh_composer=comp)[:1]
    b = ttnn.to_torch(c_t, mesh_composer=comp)[:1]
    ok = torch.equal(a, b)
    print(f"RESULT output_tile={cand}: works, combine bit-identical "
          f"{'YES' if ok else 'NO'} (max diff {(a - b).abs().max().item():.3e})", flush=True)


def timeit(fn):
    for _ in range(3):
        fn()
    ttnn.synchronize_device(mesh)
    ts = []
    for _ in range(15):
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(mesh)
        ts.append(1000 * (time.perf_counter() - t0))
    ts.sort()
    return ts[len(ts) // 2]


# --- the other way to spend less on the combine -------------------------------
# The weighted sum over E reads [1, E, 1, K] (84 MB of padding), writes a product
# of the same size, and reads it back to reduce: ~254 MB a layer for 2.6 MB of
# answer. Packing the expert axis into the row axis first makes it one matmul
# against the router weights in the shape routing already produced them --
# reads 84 MB, writes 2.6 -- and skips the permute as well.
gate_flat = ttnn.from_torch(torch.rand(1, 1, M, E), dtype=ttnn.bfloat16, **tile)


def combine_matmul(p):
    packed = ttnn.reshape(p, (1, 1, E * M, K))
    return ttnn.matmul(gate_flat, packed, compute_kernel_config=HIFI4)


base_p = down()
try:
    a = ttnn.to_torch(ttnn.sum(ttnn.multiply(base_p, ttnn.permute(
        ttnn.reshape(gate_flat, (1, 1, M, E)), (0, 3, 2, 1))), dim=1, keepdim=True),
        mesh_composer=comp)[:1]
    b = ttnn.to_torch(combine_matmul(base_p), mesh_composer=comp)[:1]
    same = torch.equal(a, b)
    d = (a - b).abs()
    print(f"RESULT combine-as-matmul: shapes {tuple(a.shape)} vs {tuple(b.shape)}  "
          f"bit-identical {'YES' if same else 'NO'}  max diff {d.max().item():.3e}  "
          f"rel {(d.max() / a.abs().max()).item():.2e}", flush=True)
    # Which one is right? Both are approximations, so compare each against the
    # exact weighted sum computed in float64 from the *device's own* operands --
    # bf16 converts to float64 exactly, so the reference has no error of its own.
    # The sum form rounds all E products to bf16 before adding; the matmul
    # accumulates them in fp32. That should make the matmul form the more
    # accurate of the two, and this is where that gets checked rather than
    # argued.
    p_ref = ttnn.to_torch(base_p, mesh_composer=comp)[:1].double()
    g_ref = ttnn.to_torch(gate_flat, mesh_composer=comp)[:1].double()
    exact = (p_ref[0, :, 0, :] * g_ref[0, 0, 0, :, None]).sum(0)
    for label, got in (("sum   ", a), ("matmul", b)):
        err = (got.double().reshape(-1) - exact).abs()
        rel = (err / exact.abs().clamp_min(1e-30)).max().item()
        print(f"RESULT   {label} vs float64 exact: max abs {err.max().item():.3e}  "
              f"mean abs {err.mean().item():.3e}  max rel {rel:.3e}", flush=True)
    print(f"RESULT   sum form    {timeit(lambda: combine(base_p)):7.3f} ms   "
          f"matmul form {timeit(lambda: combine_matmul(base_p)):7.3f} ms", flush=True)
except Exception as exc:                                     # noqa: BLE001
    print(f"RESULT combine-as-matmul rejected: {type(exc).__name__}: "
          f"{(str(exc) or repr(exc)).splitlines()[0][:160]}", flush=True)

print(f"RESULT default   down {timeit(lambda: down()):7.3f} ms   "
      f"down+combine {timeit(lambda: combine(down())):7.3f} ms", flush=True)
ttnn.close_mesh_device(mesh)
