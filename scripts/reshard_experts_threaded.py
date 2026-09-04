"""Re-shard the MoE expert stacks, several tensors at a time, in one process.

    uv run python scripts/reshard_experts_threaded.py [threads]     default 8

Conversion is CPU bound and single-threaded: ~83 s a layer, one core of the
host's forty-eight. Two things rule out the obvious fix of running several
processes: the first one to touch ttnn takes the device lock
(`CHIP_IN_USE_0_PCIe`) and holds it for its lifetime, so the rest queue -- two
concurrent layers measured 158 s against 166 s serial, i.e. no overlap at all.

Threads in one process share that single lock and still get real parallelism,
because the work is in numpy and in nanobind C++ which release the GIL: four
tensors in four threads took 67.5 s against 127 s serial, 1.89x.

Where the time goes, per expert tensor (838 M elements):

    dequantise 18.3 s   layout 0.3 s   from_torch(bfloat4_b) 12.6 s   dump 0.2 s

Resumable: only tensors still on the old axis are converted.
"""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import ttnn

from ttrunner_qwen38_flash_next.gguf.reader import GGUFModel
from ttrunner_qwen38_flash_next.reference.weights import WeightStore
from ttrunner_qwen38_flash_next.tt.convert import (
    NEEDS_ALL_REDUCE, SHARD_DIM, device_layout, prescale_factor,
)
from ttrunner_qwen38_flash_next.tt.plan import plan_for

GGUF = os.environ.get("TTRUNNER_GGUF_DIR",
                      str(Path.home() / "models/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS"))
CACHE = Path(os.environ.get("TTRUNNER_TT_CACHE", str(Path.home() / "models/qwen38-tt-cache")))
MANIFEST = CACHE / "manifest.json"
N_DEV = 4
DTYPES = {"bfloat8_b": ttnn.bfloat8_b, "bfloat4_b": ttnn.bfloat4_b,
          "bfloat16": ttnn.bfloat16, "float32": ttnn.float32}

lock = threading.Lock()
done_count = 0


def convert_one(args):
    global done_count
    name, gguf = args
    entry = plan_for(name)
    dim = SHARD_DIM[entry.shard]
    store = WeightStore(gguf, cache_bytes=1 << 28, row_cache_bytes=1 << 28)
    info = store.info(name)
    t = store.get_row_range(name, 0, info.torch_shape[0])
    scale = prescale_factor(name)
    if scale != 1.0:
        t = t * scale
    t = device_layout(name, t)
    paths = [CACHE / f"{name}.dev{i}.tensorbin" for i in range(N_DEV)]
    for piece, path in zip(torch.chunk(t, N_DEV, dim=dim), paths):
        host = ttnn.from_torch(piece.contiguous(), dtype=DTYPES[entry.dtype],
                               layout=ttnn.TILE_LAYOUT)
        ttnn.dump_tensor(str(path), host)
        del host
    del t
    record = {
        "files": [p.name for p in paths],
        "dtype": entry.dtype,
        "shard": entry.shard.name,
        "shard_dim": dim,
        "all_reduce": entry.shard in NEEDS_ALL_REDUCE,
        "shape": list(info.torch_shape),
        "bytes": sum(p.stat().st_size for p in paths),
    }
    with lock:
        done_count += 1
        print(f"RESULT [{done_count}] {name}", flush=True)
    return name, record


def main() -> None:
    threads = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    manifest = json.loads(MANIFEST.read_text())
    tensors = manifest["tensors"]
    todo = [n for n, r in tensors.items()
            if n.endswith(("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"))
            and r.get("shard") != "EXPERT"]
    print(f"RESULT {len(todo)} tensors to convert on {threads} threads", flush=True)
    if not todo:
        return
    gguf = GGUFModel.from_dir(GGUF)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for name, record in pool.map(convert_one, [(n, gguf) for n in todo]):
            tensors[name] = record
    MANIFEST.write_text(json.dumps(manifest, indent=1))
    print(f"RESULT done: {len(todo)} tensors in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
