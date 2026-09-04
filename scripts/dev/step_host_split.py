"""Split the traced decode step into host preparation and device replay.

    uv run python scripts/dev/step_host_split.py

The component ablations account for 113.5 ms of a 146.2 ms step -- attention
59.6, MoE 42.9, PLE 2.8, all_reduce 3.8, hyper-connection mixing 2.5, reinject
1.9 -- and every attempt to find the remaining ~33 ms inside the model has come
back small. So look outside it.

`TracedDecoder.step` is two things: `_fill_inputs`, which runs on the host and
writes this token's inputs into the bound buffers, and `execute_trace`, which is
the device. `_fill_inputs` calls `model.embed(tokens)` and the PLE n-gram
lookup, and both of those gather rows out of **host** memory -- token_embd and a
51 B-parameter n-gram table, served by `WeightStore` from mmap'd GGUF. A random
gather into that is not obviously cheap, and none of it is on the device at all,
so no device-side ablation could ever have seen it.

Times each half separately, then the pieces of the host half.
"""
import sys
import time

import ttnn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _device_model import open_model                                # noqa: E402

ITERS = 30

mesh, cfg, m = open_model(max_seq_len=4096)
from ttrunner_qwen38_flash_next.tt.traced import TracedDecoder       # noqa: E402

state = m.new_state(batch=1)
dec = TracedDecoder(m, state)
dec.reset()
dec.step([1000])
ttnn.synchronize_device(mesh)


def med(fn, n=ITERS, sync=True):
    xs = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        if sync:
            ttnn.synchronize_device(mesh)
        xs.append(1000 * (time.perf_counter() - t0))
    xs.sort()
    return xs[len(xs) // 2]


whole = med(lambda: dec.step([1000]))
print(f"RESULT whole step                {whole:7.2f} ms", flush=True)

# The device half on its own: replay without touching the bound buffers.
replay = med(lambda: ttnn.execute_trace(mesh, dec.trace_id, cq_id=dec.cq_id, blocking=True))
print(f"RESULT execute_trace only        {replay:7.2f} ms", flush=True)

# The host half on its own -- no device work at all, so no sync.
fill = med(lambda: dec._fill_inputs([1000], state), sync=False)
print(f"RESULT _fill_inputs only         {fill:7.2f} ms   (host, no device)", flush=True)
print(f"RESULT  -> host {fill:.2f} + device {replay:.2f} = {fill + replay:.2f} "
      f"against {whole:.2f} measured", flush=True)

# And inside the host half.
emb = med(lambda: m.embed([1000]), sync=False)
print(f"RESULT   model.embed             {emb:7.2f} ms   (host gather from token_embd)",
      flush=True)

if "ngram" in (m.bound or {}):
    print("RESULT   ngram buffer is bound -- the PLE lookup runs every step", flush=True)
    try:
        ng = med(lambda: m.ngram_rows(state) if hasattr(m, "ngram_rows") else None, sync=False)
        print(f"RESULT   ngram rows            {ng:7.2f} ms", flush=True)
    except Exception as exc:                                        # noqa: BLE001
        print(f"RESULT   ngram rows: {type(exc).__name__} -- name it by hand: "
              f"{[k for k in (m.bound or {})]}", flush=True)
else:
    print(f"RESULT   bound buffers: {sorted((m.bound or {}).keys())}", flush=True)

rope = med(lambda: m.rope([44]), sync=False)
print(f"RESULT   model.rope              {rope:7.2f} ms   (host cos/sin build)", flush=True)

print(f"RESULT verdict: the host half is {100 * fill / whole:.1f}% of the step",
      flush=True)

ttnn.close_mesh_device(mesh)
