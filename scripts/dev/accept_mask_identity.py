"""Does masking g_exp/beta really make a rejected step the identity?

The whole single-trace speculation design rests on it: a captured trace advances
the state by exactly its k, so partial acceptance has to be expressible as data.
The claim is that `g = 1, beta = 0` leaves `state = state * g + k^T delta`
untouched. This checks it against the only thing that matters -- running fewer
tokens.
"""
import sys
import torch, ttnn
sys.path.insert(0, "/home/twjang/twtest/scripts/dev")
from _device_model import open_model

mesh, cfg, m = open_model(max_seq_len=2048)
print(f"RESULT use_indexer={m.use_indexer}", flush=True)
TOK = [1000, 262, 553, 91]

def run(tokens, accept):
    st = m.new_state(batch=1)
    out = m.step_n(list(tokens), st, accept=accept)
    rec = [ttnn.to_torch(st.layers[i].recurrent, mesh_composer=m.compose)[0:1].clone()
           for i in range(cfg.num_layers) if st.layers[i].recurrent is not None]
    conv = [ttnn.to_torch(c, mesh_composer=m.compose)[0:1].clone()
            for i in range(cfg.num_layers) if st.layers[i].conv is not None
            for c in st.layers[i].conv]
    return ttnn.to_torch(out, mesh_composer=m.compose)[0:1].clone(), rec, conv

# The decisive control: same k, same graph, same accepted prefix -- only the
# *rejected* tokens differ. If masking works the state must be bit-identical,
# because those steps are supposed to be the identity. Comparing against a k=2
# run instead would confound this with the k=2-vs-k=4 matmul blocking, which is
# a different question.
o_a, r_a, c_a = run([1000, 262, 553, 91], [1.0, 1.0, 0.0, 0.0])
o_b, r_b, c_b = run([1000, 262, 7777, 31337], [1.0, 1.0, 0.0, 0.0])
worst = max((a.to(torch.float64) - b.to(torch.float64)).abs().max().item()
            for a, b in zip(r_a, r_b))
scale = max(b.to(torch.float64).abs().max().item() for b in r_b)
print(f"RESULT rejected tokens changed, state diff: {worst:.4e} "
      f"(values to {scale:.3e})", flush=True)
d_out = (o_a[..., :2, :].to(torch.float64) - o_b[..., :2, :].to(torch.float64)).abs().max().item()
print(f"RESULT accepted rows differ by: {d_out:.4e}", flush=True)

# And the unmasked control, to show the tokens really do matter without the mask.
o_c, r_c, _ = run([1000, 262, 7777, 31337], None)
o_d, r_d, _ = run([1000, 262, 553, 91], None)
unmasked = max((a.to(torch.float64) - b.to(torch.float64)).abs().max().item()
               for a, b in zip(r_c, r_d))
print(f"RESULT without the mask the same change moves the state by: {unmasked:.4e}",
      flush=True)
cworst = max((a.to(torch.float64) - b.to(torch.float64)).abs().max().item()
             for a, b in zip(c_a, c_b))
cscale = max(b.to(torch.float64).abs().max().item() for b in c_b)
print(f"RESULT conv ring, rejected tokens changed: {cworst:.4e} "
      f"(values to {cscale:.3e}) -- {len(c_a)} ring columns", flush=True)
print("RESULT conv ring is " + ("CLEAN (nothing to fix)" if cworst == 0.0
                                else "POLLUTED -- needs rebuilding from the columns"),
      flush=True)
print("RESULT verdict: " + (
    "masking IS the identity -- rejected tokens have no effect"
    if worst == 0.0 else
    f"NOT exact -- rejected tokens still move the state by {worst:.2e}"), flush=True)
ttnn.close_mesh_device(mesh)
