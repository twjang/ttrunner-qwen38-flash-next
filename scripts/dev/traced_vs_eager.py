"""Does a captured trace reproduce the eager step, token for token?

    uv run python scripts/dev/traced_vs_eager.py [n_tokens]      (default 12)
    TWTEST_MAX_SEQ=8192 ... to exercise the QSA selection as well

The project's standing gate for anything that touches the step. A trace replays
a recorded graph against the addresses it captured, so a step that reads
anything it wrote outside the capture -- or writes to the device inside it --
diverges silently, and the failure mode is plausible-looking text.
"""
import os
import sys

import ttnn

from _device_model import open_model, tokenizer

from ttrunner_qwen38_flash_next.tt.traced import TracedDecoder

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
SEQ = int(os.environ.get("TWTEST_MAX_SEQ", "8192"))

mesh, cfg, m = open_model(max_seq_len=SEQ)
tok = tokenizer(cfg)
print(f"RESULT max_seq_len {SEQ}  indexer {'on' if m.use_indexer else 'off'}", flush=True)
prompt = tok.encode("The capital of France is")


def run(step, state=None):
    out = []
    h = None
    for t in prompt:
        h = step([t])
    for _ in range(N):
        nxt = m.greedy_tokens(h)[0]
        out.append(nxt)
        h = step([nxt])
    return out


st = m.new_state(batch=1)
eager = run(lambda ids: m.step(ids, st))
del st
print(f"RESULT eager  {eager} {tok.decode(eager)!r}", flush=True)

st = m.new_state(batch=1)
dec = TracedDecoder(m, st)
dec.reset()
traced = run(dec.step)
print(f"RESULT traced {traced} {tok.decode(traced)!r}", flush=True)
print(f"RESULT {'MATCH' if eager == traced else 'DIFFER'}", flush=True)
ttnn.close_mesh_device(mesh)
