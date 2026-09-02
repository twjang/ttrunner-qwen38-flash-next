"""How far are the device weights from the GGUF they were converted from?

    uv run python scripts/dev/weight_fidelity.py [layer]          (default 0)

The decode path agrees with the float32 oracle on only a quarter of tokens, and
it is already 52 % away after layer 0 -- too early for accumulation to explain.
Either a block is wrong or the weights are. This checks the weights: every
tensor of one layer, device against the dequantised GGUF, gathered along
whatever axis the manifest says it was sharded on.

bfloat4_b is a second quantisation on top of the checkpoint's own 4-bit, so
~6 % relative on a weight is expected there; bfloat8_b should be ~0.4 % and
bfloat16 ~0.4 %. What matters is a tensor that is far worse than its dtype
allows, or one whose shape does not line up at all.
"""
import sys

import torch
import ttnn

from _device_model import open_model

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 0
mesh, cfg, m = open_model(preload=False)

for name, record in sorted(m.w.entries.items()):
    if not name.startswith(f"blk.{LAYER}."):
        continue
    if "exps" in name:                      # far too large to bring back whole
        continue
    dim = record["shard_dim"]
    dev = m.w.get(name)
    if dim is None:
        got = ttnn.to_torch(dev, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        got = got[: got.shape[0] // m.w.n_dev]
    else:
        got = ttnn.to_torch(dev, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=dim))
    want = m.host.get(name).float().squeeze()
    got = got.float().squeeze()
    if got.numel() != want.numel():
        print(f"RESULT {name:44s} SIZE {tuple(got.shape)} vs {tuple(want.shape)}", flush=True)
        continue

    # ttnn.linear wants [in, out] where torch stores [out, in], so the converter
    # transposes. Try both and report the orientation that lines up -- an
    # rms of ~141 % (sqrt 2) means the two are simply uncorrelated, which is a
    # layout mismatch in the comparison, not a bad weight.
    scale = max(want.abs().max().item(), 1e-9)
    rms_scale = max(want.pow(2).mean().sqrt().item(), 1e-9)
    best = None
    for label, cand in (
        ("as-is", got.reshape(want.shape) if got.numel() == want.numel() else None),
        ("transposed", got.reshape(want.T.shape).T if want.ndim == 2 else None),
    ):
        if cand is None:
            continue
        d = (cand - want)
        entry = (d.pow(2).mean().sqrt().item() / rms_scale, label,
                 100 * d.abs().max().item() / scale)
        if best is None or entry[0] < best[0]:
            best = entry
    rms, label, rel = best
    print(
        f"RESULT {name:44s} dtype {str(dev.dtype).split('.')[-1]:12s} shard {str(dim):4s} "
        f"{label:11s} max {rel:7.2f}%  rms {100 * rms:7.2f}%",
        flush=True,
    )
ttnn.close_mesh_device(mesh)
