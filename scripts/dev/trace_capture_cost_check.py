"""What does capturing -- and re-capturing -- the decode trace cost?

    uv run python scripts/dev/trace_capture_cost_check.py [repeats]

An agent workload wants chunked prefill *and* the traced decode step: ingestion
at 7.2 ms/token and generation at 177 rather than 588. Handoff 5.2 says they
cannot both be on, because a prefill that allocates while a trace is live
corrupts the replay. One way round that is to not have a trace live during a
prefill: release it, prefill, capture again.

Worth building only if a capture is cheap against what it saves. A 200-token
generation saves 200 x (588 - 177) ms = 82 s, so nearly any capture cost pays --
but "nearly" needs a number, and repeated capture/release also has to not leak or
fragment, which is what the repeats are for.
"""
import sys
import time

import ttnn

from _device_model import open_model

REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 4

mesh, cfg, m = open_model(max_seq_len=4096, trace_region_bytes=128 << 20)
from ttrunner_qwen38_flash_next.tt.traced import TracedDecoder  # noqa: E402

state = m.new_state(batch=1)
for i in range(REPEATS):
    t0 = time.perf_counter()
    dec = TracedDecoder(m, state)
    dec.reset()
    ttnn.synchronize_device(mesh)
    cap = time.perf_counter() - t0

    t1 = time.perf_counter()
    for _ in range(5):
        dec.step([1000])
    ttnn.synchronize_device(mesh)
    step = (time.perf_counter() - t1) / 5

    t2 = time.perf_counter()
    dec.release()
    ttnn.synchronize_device(mesh)
    relt = time.perf_counter() - t2

    # a prefill in the gap, which is the whole point of releasing
    t3 = time.perf_counter()
    st2 = m.new_state(batch=1)
    m.prefill([1000] * 128, st2)
    ttnn.synchronize_device(mesh)
    pf = time.perf_counter() - t3

    print(f"RESULT round {i}: capture {1000 * cap:8.1f} ms   step {1000 * step:7.1f} ms   "
          f"release {1000 * relt:7.1f} ms   prefill-in-gap {1000 * pf:7.1f} ms", flush=True)

ttnn.close_mesh_device(mesh)
