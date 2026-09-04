"""What the decode step's *output side* costs, which the ablations never measured.

    uv run python scripts/dev/output_side_cost.py

`TracedDecoder` captures `model.step(...)` and nothing else -- `logits` and
`greedy_tokens` are warmed before the capture region so their buffers exist, but
they are replayed by nobody. So the 110.8 ms the ablation harness reports is the
traced step alone, and the engine pays the LM head, the argmax and the host round
trip on top of it, every token.

That is also where most of the ~16 ms unattributed in the component budget has
to live, since every other part of `step` is accounted for.

Prices, separately:
  - `logits()`        the full [B, vocab] gather off four devices
  - `greedy_tokens()` the device-side per-shard max/argmax the engine prefers
  - the traced step, for scale

The LM head is 248320 x 2560. At bfloat4_b that is ~357 MB, ~89 MB a device, so
a bandwidth-bound head should cost ~0.23 ms at 388 GB/s. A GEMV padded to a
32-row tile wastes compute but reads the same bytes, so anything far above that
is dispatch or the host round trip, not the weights.
"""
import sys
import time

import ttnn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _device_model import open_model                                # noqa: E402

ITERS = 20

mesh, cfg, m = open_model(max_seq_len=4096)
from ttrunner_qwen38_flash_next.tt.traced import TracedDecoder       # noqa: E402

state = m.new_state(batch=1)
dec = TracedDecoder(m, state)
dec.reset()

hidden = dec.step([1000])
ttnn.synchronize_device(mesh)


def med(fn, n=ITERS):
    xs = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(mesh)
        xs.append(1000 * (time.perf_counter() - t0))
    xs.sort()
    return xs[len(xs) // 2], xs[0]


step_med, step_min = med(lambda: dec.step([1000]))
print(f"RESULT traced step          median {step_med:7.2f} ms  min {step_min:7.2f}", flush=True)

g_med, g_min = med(lambda: m.greedy_tokens(hidden))
print(f"RESULT greedy_tokens        median {g_med:7.2f} ms  min {g_min:7.2f}", flush=True)

l_med, l_min = med(lambda: m.logits(hidden))
print(f"RESULT logits (full gather) median {l_med:7.2f} ms  min {l_min:7.2f}", flush=True)


# What the engine actually does per token: replay, then argmax.
def engine_round():
    h = dec.step([1000])
    m.greedy_tokens(h)


e_med, e_min = med(engine_round)
print(f"RESULT step + greedy        median {e_med:7.2f} ms  min {e_min:7.2f}", flush=True)
print(f"RESULT output side is {e_med - step_med:6.2f} ms on top of the step "
      f"({100 * (e_med - step_med) / e_med:.1f}% of the token)", flush=True)

vocab, hid = cfg.vocab_size, cfg.hidden_size
head_bytes = vocab * hid * 0.5625 / 4          # bfloat4_b, per device
print(f"RESULT lm_head {vocab}x{hid}: {head_bytes / 1e6:.0f} MB/device -> "
      f"{head_bytes / 388e9 * 1e3:.3f} ms at 388 GB/s", flush=True)
print(f"RESULT so greedy_tokens is {g_med / (head_bytes / 388e9 * 1e3):.0f}x its "
      f"own bandwidth floor", flush=True)

ttnn.close_mesh_device(mesh)
