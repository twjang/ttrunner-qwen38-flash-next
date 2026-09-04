"""Re-shard the MoE expert stacks, several layers at a time.

    uv run python scripts/reshard_experts_parallel.py [workers]      default 8

The sequential version takes ~83 s a layer and 66 minutes for the model, and
`top` says why: one core busy out of forty-eight. Conversion never touches the
device -- `ttnn.from_torch` without `device=` builds a host tensor and
`dump_tensor` writes it -- so it parallelises over layers with nothing shared but
the manifest, and each worker writes its own for a merge at the end.

Only layers still on the old axis are converted, so this resumes an interrupted
run rather than redoing it.
"""
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

GGUF = os.environ.get("TTRUNNER_GGUF_DIR",
                      str(Path.home() / "models/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS"))
CACHE = Path(os.environ.get("TTRUNNER_TT_CACHE", str(Path.home() / "models/qwen38-tt-cache")))
MANIFEST = CACHE / "manifest.json"


def todo_layers() -> list[int]:
    tensors = json.loads(MANIFEST.read_text())["tensors"]
    done = {int(k.split(".")[1]) for k, v in tensors.items()
            if k.endswith("ffn_gate_exps.weight") and v.get("shard") == "EXPERT"}
    return [l for l in range(48) if l not in done]


def work(job):
    wid, layers = job
    from ttrunner_qwen38_flash_next.tt.convert import convert

    pattern = rf"^blk\.({'|'.join(str(l) for l in layers)})\.ffn_(gate|up|down)_exps\.weight$"
    stats = convert(GGUF, CACHE, n_dev=4, only=pattern, force=True,
                    manifest_name=f"manifest.w{wid}.json", progress=None)
    return wid, len(layers), stats.bytes_written


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    layers = todo_layers()
    if not layers:
        print("RESULT nothing to do", flush=True)
        return
    chunks = [layers[i::workers] for i in range(workers)]
    chunks = [c for c in chunks if c]
    print(f"RESULT {len(layers)} layers over {len(chunks)} workers", flush=True)

    t0 = time.time()
    with mp.get_context("spawn").Pool(len(chunks)) as pool:
        for wid, n, nbytes in pool.imap_unordered(work, list(enumerate(chunks))):
            print(f"RESULT worker {wid} done: {n} layers, {nbytes / 1e9:.1f} GB "
                  f"[{time.time() - t0:.0f}s]", flush=True)

    # fold the per-worker manifests into the real one
    manifest = json.loads(MANIFEST.read_text())
    merged = 0
    for part in sorted(CACHE.glob("manifest.w*.json")):
        for name, rec in json.loads(part.read_text())["tensors"].items():
            manifest["tensors"][name] = rec
            merged += 1
        part.unlink()
    MANIFEST.write_text(json.dumps(manifest, indent=1))
    print(f"RESULT merged {merged} entries in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
