"""Does capturing `step_n` on a worker thread work?

    uv run python scripts/dev/traced_step_n_thread.py [k]         (default 2)

`TTEngine(speculate=k)` hangs the device when it captures `step_n` inside its
device thread, and the boards then need `tt-smi -r`. Two of the three suspects
are already ruled out: `traced_step_n_check.py` performs the same post-capture
state rewind on the main thread and works, and the engine forces `use_trace` off
when speculating, so two live captures is not it either.

That leaves the thread. `TracedDecoder` captures the single-token step on that
same device thread and is fine, so if this reproduces, the answer is oddly
specific -- capturing *this* graph off the main thread -- and worth reporting
upstream rather than working around.

Bounded on purpose: a watchdog gives up rather than hanging the session.
"""
import sys
import threading
import time

import ttnn

from _device_model import open_model, synthetic_prompt

from ttrunner_qwen38_flash_next.tt.traced import TracedDecoder, TracedStepN

K = int(sys.argv[1]) if len(sys.argv) > 1 else 2
BUDGET = 240.0

mesh, cfg, m = open_model(max_seq_len=512)
prompt = synthetic_prompt(8 + K)
done = threading.Event()
out = {}


def worker():
    try:
        st = m.new_state(batch=1)
        print("RESULT worker: warming", flush=True)
        for t in prompt[:8]:
            m.step([t], st)
        # The engine captures *both*: a decoder for ordinary rounds and a step_n
        # for verification. One capture on this thread is fine and two on the
        # main thread are fine; this is the combination left untested.
        print("RESULT worker: capturing decoder", flush=True)
        dec = TracedDecoder(m, st)
        dec.reset()
        print("RESULT worker: decoder captured", flush=True)
        for t in prompt[:8]:
            dec.step([t])
        print("RESULT worker: capturing step_n", flush=True)
        tr = TracedStepN(m, st, K)
        print("RESULT worker: captured", flush=True)
        for layer in st.layers:
            for name in ("recurrent", "conv", "ple_conv", "keys", "values"):
                buf = getattr(layer, name)
                if buf is None:
                    continue
                for entry in (buf if isinstance(buf, list) else [buf]):
                    ttnn.copy(
                        ttnn.zeros(list(entry.shape), dtype=entry.dtype,
                                   layout=entry.layout, device=mesh),
                        entry,
                    )
            layer.conv_step = 0
            layer.ple_step = 0
        st.positions = [0]
        st.histories = [[]]
        print("RESULT worker: rewound", flush=True)
        for t in prompt[:8]:
            m.step([t], st)
        t0 = time.perf_counter()
        tr.step_n(prompt[8 : 8 + K])
        ttnn.synchronize_device(mesh)
        out["ms"] = 1000 * (time.perf_counter() - t0)
        print(f"RESULT worker: replayed in {out['ms']:.1f} ms", flush=True)
        tr.release()
        dec.release()
    except Exception as exc:
        out["error"] = " ".join(str(exc).split())[:200]
        print(f"RESULT worker: FAILED {out['error']}", flush=True)
    finally:
        done.set()


th = threading.Thread(target=worker, name="device", daemon=True)
th.start()
if not done.wait(BUDGET):
    print(f"RESULT HUNG -- no completion in {BUDGET:.0f}s; this reproduces the engine's hang",
          flush=True)
    print("RESULT the process is left for inspection; the board will need tt-smi -r", flush=True)
else:
    print(f"RESULT completed on a worker thread: {out}", flush=True)
    ttnn.close_mesh_device(mesh)
