"""Does `restore` really rewind the state a speculative round advanced?

    uv run python scripts/dev/snapshot_check.py

Verifying k drafts advances the DeltaNet recurrence and the convolution rings by
all k tokens, and a recurrence cannot be truncated back the way a K/V cache can.
So a rejected draft has to be rolled back, and "rolled back" has to mean the
sequence continues exactly as if the draft had never been run.

The test: generate a continuation, then generate it again with a wrong draft run
and rolled back in between. The two must be identical, token for token.
"""
import ttnn

from _device_model import open_model, synthetic_prompt

mesh, cfg, m = open_model(max_seq_len=512)
prompt = synthetic_prompt(8)
N = 6


def cont(st, first):
    """Feed `first`, then generate. Depends only on tokens and the state, never
    on a device tensor that the discarded draft might have reallocated."""
    out = [first]
    h = m.step([first], st)
    for _ in range(N - 1):
        t = m.greedy_tokens(h)[0]
        out.append(t)
        h = m.step([t], st)
    return out


st = m.new_state(batch=1)
h = None
for t in prompt:
    h = m.step([t], st)
first = m.greedy_tokens(h)[0]
clean = cont(st, first)
del st

st = m.new_state(batch=1)
for t in prompt:
    m.step([t], st)
snap = m.snapshot(st)
# a draft that will be thrown away: four arbitrary tokens through step_n
m.step_n([4242, 1337, 999, 12345], st)
print(f"RESULT after the discarded draft, position {st.positions[0]}", flush=True)
m.restore(st, snap)
print(f"RESULT after restore, position {st.positions[0]}   "
      f"history {len(st.histories[0])}", flush=True)
rolled = cont(st, first)

print(f"RESULT clean   {clean}", flush=True)
print(f"RESULT rolled  {rolled}", flush=True)
print(f"RESULT {'MATCH' if clean == rolled else 'DIFFER'}", flush=True)
ttnn.close_mesh_device(mesh)
