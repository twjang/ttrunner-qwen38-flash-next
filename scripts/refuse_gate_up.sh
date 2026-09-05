#!/usr/bin/env bash
# Drop the fused gate|up tensors and rebuild them from the current gate/up shards.
#
# `fuse_expert_gate_up.py` skips a layer whose fused files already exist, which is
# what makes it resumable -- and what makes it a no-op after a re-shard, when the
# fused tensor is the only thing still on the old axis. Every measurement would
# then be taken against a gate|up that does not match its own gate and up.
set -euo pipefail
CACHE="${TTRUNNER_TT_CACHE:-$HOME/models/qwen38-tt-cache}"
n=$(ls "$CACHE"/*ffn_gateup_exps.weight.dev*.tensorbin 2>/dev/null | wc -l)
echo "RESULT removing $n stale fused gate|up shards"
rm -f "$CACHE"/*ffn_gateup_exps.weight.dev*.tensorbin
python3 - "$CACHE" <<'PY'
import json, sys, pathlib
man = pathlib.Path(sys.argv[1]) / "manifest.json"
d = json.loads(man.read_text())
gone = [k for k in d["tensors"] if k.endswith("ffn_gateup_exps.weight")]
for k in gone:
    del d["tensors"][k]
man.write_text(json.dumps(d, indent=1))
print(f"RESULT dropped {len(gone)} manifest entries")
PY
uv run python scripts/fuse_expert_gate_up.py "$CACHE"
