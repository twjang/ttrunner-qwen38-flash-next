#!/usr/bin/env python3
"""Regenerate src/twtest/gguf/_codebooks.py from ggml-common.h.

The IQ3_S grid is 512 uint32 constants and IQ4_NL's codebook is 16 int8s.
Hand-copying them risks a silent single-digit corruption that no shape check
would catch, so they are parsed from upstream instead.

Usage: python3 scripts/gen_codebooks.py [path/to/ggml-common.h]
Without an argument the header is fetched from llama.cpp master.
"""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

SOURCE_URL = "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-common.h"
OUT = Path(__file__).resolve().parent.parent / "src" / "twtest" / "gguf" / "_codebooks.py"


def parse_table(src: str, name: str) -> list[int]:
    match = re.search(
        r"GGML_TABLE_BEGIN\(\s*\w+\s*,\s*" + name + r"\s*,\s*(\d+)\s*\)(.*?)GGML_TABLE_END\(\)",
        src,
        re.S,
    )
    if match is None:
        raise SystemExit(f"table {name!r} not found in ggml-common.h")
    declared = int(match.group(1))
    values = [int(v, 0) for v in re.findall(r"(0x[0-9a-fA-F]+|-?\d+)", match.group(2))]
    if len(values) != declared:
        raise SystemExit(f"{name}: parsed {len(values)} values but header declares {declared}")
    return values


def main() -> int:
    if len(sys.argv) > 1:
        src = Path(sys.argv[1]).read_text()
    else:
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as r:
            src = r.read().decode()

    grid = parse_table(src, "iq3s_grid")
    kvalues = parse_table(src, "kvalues_iq4nl")

    lines = [
        '"""Quantization codebooks extracted verbatim from ggml-common.h (llama.cpp, MIT).',
        "",
        "Generated - do not edit by hand. Regenerate with scripts/gen_codebooks.py.",
        '"""',
        "",
        "import numpy as np",
        "",
        "# IQ4_NL / IQ4_XS 16-entry non-linear codebook.",
        f"KVALUES_IQ4NL = np.array({kvalues}, dtype=np.int8)",
        "",
        "# IQ3_S 512-entry grid; each uint32 packs four uint8 magnitudes.",
        "IQ3S_GRID_PACKED = np.array([",
    ]
    for i in range(0, len(grid), 8):
        lines.append("    " + ", ".join(f"0x{v:08x}" for v in grid[i : i + 8]) + ",")
    lines += [
        "], dtype=np.uint32)",
        "",
        "# Unpacked to (512, 4) uint8 for vectorised gathers.",
        "IQ3S_GRID = IQ3S_GRID_PACKED.view(np.uint8).reshape(512, 4)",
        "",
    ]
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT} ({len(grid)} grid entries, {len(kvalues)} codebook entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
