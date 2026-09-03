"""Does a sequence decode the same in a batch as it does alone?

    uv run python scripts/dev/batch_equivalence_check.py [batch ...]  (default 8 32 64)

The README claims batch 64 is "bit-exact vs single-sequence" and nothing in the
tree checks it. It matters more than a slogan now: `sparse_program_config`
switches on `per_core_M > 1`, which for decode means **batch > 32**, because
`ttnn.sparse_matmul` silently loses rows past the first 32-row tile when K spans
more than one block (handoff 5.8). Before that fix batch 64 was decoding its
upper rows through the broken configuration.

Every row is given the same prompt, so all rows must agree with each other *and*
with a batch-1 run of that prompt. Disagreement between rows localises to the
row axis; agreement between rows but not with batch 1 points at something the
whole batch shares.
"""
import sys

import ttnn

from _device_model import open_model, synthetic_prompt

BATCHES = [int(x) for x in sys.argv[1:]] or [8, 32, 64]
PROMPT, GEN = 8, 12

mesh, cfg, m = open_model(max_seq_len=512)
prompt = synthetic_prompt(PROMPT)


def decode(batch):
    st = m.new_state(batch=batch)
    for t in prompt:
        hidden = m.step([t] * batch, st)
    rows = [[] for _ in range(batch)]
    toks = m.greedy_tokens(hidden)
    for _ in range(GEN):
        for i in range(batch):
            rows[i].append(int(toks[i]))
        hidden = m.step([int(toks[i]) for i in range(batch)], st)
        toks = m.greedy_tokens(hidden)
    return rows


single = decode(1)[0]
print(f"RESULT batch 1 reference: {single}", flush=True)
print(f"RESULT {'batch':>6} {'rows == row 0':>14} {'row 0 == batch 1':>18} "
      f"{'rows matching batch 1':>22}", flush=True)
for batch in BATCHES:
    try:
        rows = decode(batch)
        same_as_first = sum(1 for r in rows if r == rows[0])
        match_single = sum(1 for r in rows if r == single)
        flag = "" if match_single == batch else "   <-- DIVERGES"
        print(f"RESULT {batch:6d} {same_as_first:9d}/{batch:<4} "
              f"{str(rows[0] == single):>18} {match_single:15d}/{batch:<4}{flag}", flush=True)
        if match_single != batch:
            bad = next(i for i, r in enumerate(rows) if r != single)
            print(f"RESULT   first divergent row {bad}: {rows[bad]}", flush=True)
    except Exception as exc:
        print(f"RESULT {batch:6d}  FAILED {' '.join(str(exc).split())[:140]}", flush=True)
ttnn.close_mesh_device(mesh)
