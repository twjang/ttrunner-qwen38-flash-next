"""How many experts does the router actually keep, per row, on real text?

    uv run python scripts/dev/kept_expert_census.py [n_tokens]

`moe_block` admits ties -- `keep = probs >= threshold`, where the threshold is
the top_k-th largest probability -- so a row keeps *at least* top_k experts and
sometimes more. The docstring estimates ~11.5 where top_k is 10.

That number decides whether the experts can be sharded on the intermediate axis
instead of the expert axis. Today each device holds a quarter of the experts and
gathers up to 10 of them, so the union across four devices covers every kept
expert. Sharding the other way makes every device select the *same* global list,
and its length is then a fixed k_sel -- so k_sel has to cover the tail, not the
mean. This measures the tail on real activations rather than assuming it.
"""
import sys
from collections import Counter

import torch
import ttnn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _device_model import open_model, tokenizer                      # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 48

mesh, cfg, m = open_model(max_seq_len=512)
import ttrunner_qwen38_flash_next.tt.moe as moe                      # noqa: E402

tok = tokenizer(cfg)
_enc = tok.encode(
    "The history of computing hardware covers the developments from early simple "
    "devices to aid calculation to modern day computers. Machine learning models "
    "route tokens to a small subset of experts, which is what makes a mixture of "
    "experts cheap to run at inference time."
)
PROMPT = list(getattr(_enc, "ids", _enc))[:N]
print(f"RESULT {len(PROMPT)} tokens, top_k {cfg.num_experts_per_tok}, "
      f"{cfg.num_experts} experts", flush=True)

hist = Counter()
per_dev_hist = Counter()
n_dev = mesh.get_num_devices()
e_local = cfg.num_experts // n_dev
real_block = moe.moe_block


def counting_block(x, router_w, gate_w, up_w, down_w, top_k, E, K, I):
    logits = ttnn.linear(x, router_w, compute_kernel_config=moe.HIFI4)
    probs = ttnn.softmax(logits, dim=-1, compute_kernel_config=moe.HIFI4)
    values, _ = ttnn.topk(probs, k=top_k, dim=-1, largest=True, sorted=True)
    v = list(values.shape)
    threshold = ttnn.slice(values, (0, 0, 0, top_k - 1), (v[0], v[1], v[2], top_k))
    keep = ttnn.ge(probs, threshold, dtype=ttnn.bfloat16)
    host = ttnn.to_torch(keep, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1]
    row = host[0, 0, 0, :E]
    hist[int(row.sum().item())] += 1
    for d in range(n_dev):
        per_dev_hist[int(row[d * e_local:(d + 1) * e_local].sum().item())] += 1
    return real_block(x, router_w, gate_w, up_w, down_w, top_k, E, K, I)


moe.moe_block = counting_block
state = m.new_state(batch=1)
for t in PROMPT:
    m.step([t], state)

tot = sum(hist.values())
print(f"RESULT {tot} (token, layer) routing decisions", flush=True)
print(f"RESULT kept globally: {dict(sorted(hist.items()))}", flush=True)
mean = sum(k * n for k, n in hist.items()) / max(tot, 1)
print(f"RESULT   mean {mean:.2f}, max {max(hist)}", flush=True)
for k_sel in (10, 12, 14, 16):
    over = sum(n for k, n in hist.items() if k > k_sel)
    print(f"RESULT   a global k_sel of {k_sel:2d} would drop an expert on "
          f"{over}/{tot} = {100 * over / max(tot, 1):.3f}% of decisions", flush=True)
print(f"RESULT kept on one device today: {dict(sorted(per_dev_hist.items()))}", flush=True)
dtot = sum(per_dev_hist.values())
over10 = sum(n for k, n in per_dev_hist.items() if k > 10)
print(f"RESULT   today's per-device k_sel of 10 drops an expert on "
      f"{over10}/{dtot} = {100 * over10 / max(dtot, 1):.3f}% of (decision, device) pairs",
      flush=True)
ttnn.close_mesh_device(mesh)
