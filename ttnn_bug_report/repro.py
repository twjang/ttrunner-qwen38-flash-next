"""Alternating replays of two traces whose ops differ in shape hangs the device.

    TT_GGUF_DIR=... TT_CACHE_DIR=... python repro.py [k_a] [k_b]     default 2 4

Needs the model, because a self-contained version does not reproduce -- see
`minimal_attempt_does_not_reproduce.py` for what was tried.

What it does: builds two captures of the same 48-layer decode graph at two
different row counts (k tokens ride the batch axis), then replays A, B, A.

Expected: three replays return.
Observed:  the replay of B never returns. The process spins on one thread and
           the boards then need `tt-smi -r all`.

A watchdog bounds the run and reports rather than hanging the caller. Reaching
the hang takes a few minutes of kernel compilation first.
"""
import os
import sys
import threading
import time
from pathlib import Path

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twtest.gguf.reader import GGUFModel               # noqa: E402
from twtest.reference.config import Qwen4ExpConfig     # noqa: E402
from twtest.reference.weights import WeightStore       # noqa: E402
from twtest.tt.model import TTModel                    # noqa: E402
from twtest.tt.traced import TracedStepN               # noqa: E402
from twtest.tt.weights import TTWeights                # noqa: E402

K_A = int(sys.argv[1]) if len(sys.argv) > 1 else 2
K_B = int(sys.argv[2]) if len(sys.argv) > 2 else 4
GGUF = os.environ.get("TT_GGUF_DIR", str(Path.home() / "models/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS"))
CACHE = os.environ.get("TT_CACHE_DIR", str(Path.home() / "models/qwen38-tt-cache"))
BUDGET = 420.0
TOKENS = [1000 + ((i * 37) % 5000) for i in range(32)]


def main(out):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=384 << 20)
    try:
        gguf = GGUFModel.from_dir(GGUF)
        cfg = Qwen4ExpConfig.from_gguf(gguf.metadata)
        host = WeightStore(gguf, cache_bytes=2 << 30, row_cache_bytes=8 << 30)
        w = TTWeights(CACHE, mesh)
        model = TTModel(cfg, w, host, mesh, max_seq_len=2048, traceable_kv=True)
        model.fuse_expert_gate_up = "blk.0.ffn_gateup_exps.weight" in w
        print("[repro] model open; warming", flush=True)

        state = model.new_state(batch=1)
        for t in TOKENS[:8]:
            model.step([t], state)

        print(f"[repro] capturing A (k={K_A})", flush=True)
        a = TracedStepN(model, state, K_A)
        print(f"[repro] capturing B (k={K_B})", flush=True)
        b = TracedStepN(model, state, K_B)

        print("[repro] replaying A", flush=True)
        a.step_n(TOKENS[8 : 8 + K_A])
        print("[repro] replaying B   <-- hangs here", flush=True)
        b.step_n(TOKENS[16 : 16 + K_B])
        print("[repro] replaying A again -- NO HANG", flush=True)
        a.step_n(TOKENS[24 : 24 + K_A])
        out["ok"] = True
    except Exception as exc:
        out["error"] = " ".join(str(exc).split())[:300]
        print(f"[repro] FAILED {out['error']}", flush=True)
    finally:
        out["done"] = True


res: dict = {}
threading.Thread(target=main, args=(res,), daemon=True).start()
deadline = time.time() + BUDGET
while time.time() < deadline and not res.get("done"):
    time.sleep(1.0)
if not res.get("done"):
    print(f"[repro] HUNG -- no progress in {BUDGET:.0f}s. The last line printed says "
          "which replay blocked; the boards need `tt-smi -r all`.", flush=True)
else:
    print(f"[repro] finished: {res}", flush=True)
