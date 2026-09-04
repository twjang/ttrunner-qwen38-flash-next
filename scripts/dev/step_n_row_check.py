"""Does step_n's row i predict what a plain step at position i would?

If it does not, every speculative verify is comparing the drafter against the
wrong tokens, and the acceptance count means nothing.
"""
import sys
import torch, ttnn
sys.path.insert(0, "/home/twjang/twtest/scripts/dev")
from _device_model import open_model, tokenizer

mesh, cfg, m = open_model(max_seq_len=2048)
tok = tokenizer(cfg)
_enc = tok.encode("The history of computing hardware covers the developments from early simple devices.")
P = list(getattr(_enc, "ids", _enc))
K = 4

def plain(n):
    st = m.new_state(batch=1)
    for t in P[:-1]:
        m.step([t], st)
    toks, nxt = [], P[-1]
    for _ in range(n):
        h = m.step([nxt], st)
        nxt = int(m.greedy_tokens(h)[0])
        toks.append(nxt)
    return toks

want = plain(K)
print(f"RESULT plain steps          : {want}", flush=True)

st = m.new_state(batch=1)
for t in P[:-1]:
    m.step([t], st)
feed = [P[-1]] + want[:K - 1]          # the true continuation, so all k rows are on-path
h = m.step_n(feed, st)
g = m.greedy_tokens(h)
print(f"RESULT greedy_tokens(step_n): {list(g)[:K]}  (len {len(g)})", flush=True)
print(f"RESULT hidden shape {tuple(h.shape)}", flush=True)
same = list(g)[:K] == want
print("RESULT verdict: " + ("row i == plain step i -- verification is sound"
                            if same else
                            "MISMATCH -- greedy_tokens is not per-row on a k-row hidden"),
      flush=True)
if not same:
    lg = m.logits(h)
    t = ttnn.to_torch(lg, mesh_composer=m.compose)
    print(f"RESULT logits shape {tuple(t.shape)}; per-row argmax "
          f"{[int(t.reshape(-1, t.shape[-1])[i].argmax()) for i in range(K)]}", flush=True)
ttnn.close_mesh_device(mesh)
