"""Re-convert the shared expert's `gate|up|down` as shards.

    uv run python scripts/reshard_shexp.py

`ffn_(gate|up)_shexp` moved from `Shard.REPLICATE` to `Shard.COLUMN` and
`ffn_down_shexp` to `Shard.ROW` in plan.py. plan.py had replicated them "to
avoid a collective per layer", but `_moe_block` all-reduces the routed sum on
the next line anyway and the sharded `down`'s partial adds into that one: same
collective count, a quarter of the bytes (266 -> 66 MB a device a token).

144 tensors, ~270 MB. Like `reshard_hc_down.py` this needs no device --
`convert()` builds host tensors and `dump_tensor` writes them -- and for the
same reason it runs in one process: importing ttnn opens the UMD cluster and
takes `CHIP_IN_USE_0_PCIe`, so parallel workers queue on one lock forever.

The replicated files stay on disk. Only the manifest decides which form is
loaded, so restoring `manifest.json` from the backup this writes puts the model
back on the replicated weights without reconverting anything.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

GGUF = os.environ.get("TT_GGUF_DIR", str(Path.home() / "models/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS"))
CACHE = os.environ.get("TT_CACHE_DIR", str(Path.home() / "models/qwen38-tt-cache"))
PATTERN = r"^blk\.\d+\.ffn_(gate|up|down)_shexp\.weight$"


def main() -> None:
    cache = Path(CACHE)
    main_path = cache / "manifest.json"
    backup = cache / "manifest.pre-shexp-shard.json"
    if not backup.exists():
        shutil.copy2(main_path, backup)
        print(f"backed up manifest -> {backup.name}", flush=True)

    from ttrunner_qwen38_flash_next.tt.convert import convert

    t0 = time.time()
    stats = convert(
        GGUF, CACHE, n_dev=4,
        only=PATTERN,
        force=True,                       # the replicated files exist and stay
        manifest_name="manifest.shexp.json",
        progress=None,
    )
    print(f"converted {stats.tensors} tensors, {stats.bytes_written / 1e6:.0f} MB, "
          f"{time.time() - t0:.0f}s", flush=True)

    merged = json.loads(main_path.read_text())
    part = cache / "manifest.shexp.json"
    merged["tensors"].update(json.loads(part.read_text()).get("tensors", {}))
    part.unlink()
    main_path.write_text(json.dumps(merged, indent=1))

    want = {"ffn_gate_shexp": "COLUMN", "ffn_up_shexp": "COLUMN", "ffn_down_shexp": "ROW"}
    bad = []
    n = 0
    for k, v in merged["tensors"].items():
        for stem, shard in want.items():
            if k.endswith(f".{stem}.weight"):
                n += 1
                if v.get("shard") != shard:
                    bad.append((k, v.get("shard")))
    print(f"manifest merged: {n} shexp entries, "
          f"{'all sharded' if not bad else f'STILL WRONG: {bad[:4]}'}", flush=True)
    if bad or n != 144:
        sys.exit(1)


if __name__ == "__main__":
    main()
