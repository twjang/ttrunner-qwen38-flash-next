"""Run the OpenAI-compatible server.

    python -m ttrunner_qwen38_flash_next.server --model /path/to/UD-IQ4_XS --tokenizer /path/to/tokenizer.json

By default this serves the CPU reference engine, which is correct but slow
(~12 s/token). Pass --backend tt once the ttnn engine is available.
"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from .api import create_app

logger = logging.getLogger("ttrunner_qwen38_flash_next.server")


def build_engine(args: argparse.Namespace):
    if args.backend == "reference":
        from ..engine import ReferenceEngine
        from ..reference.loader import load

        logger.info("loading checkpoint from %s", args.model)
        model, tokenizer, config = load(args.model, args.tokenizer, cache_bytes=args.cache_bytes)
        if tokenizer is None:
            raise SystemExit("--tokenizer is required for the reference backend")
        engine = ReferenceEngine(model, tokenizer, config, max_concurrency=args.max_concurrency)
        # Both stop ids: the GGUF records only the first of the two.
        engine.stop_token_ids = tuple(t for t in (config.eos_token_id, 248044) if t is not None)
        logger.info("model ready: %d layers, %d experts", config.num_layers, config.num_experts)
        return engine

    if args.backend == "tt":
        from ..tt.engine import TTEngine

        if not args.tt_cache:
            raise SystemExit("--tt-cache is required for the tt backend")
        if not args.tokenizer:
            raise SystemExit("--tokenizer is required for the tt backend")
        logger.info("opening mesh and loading device weights from %s", args.tt_cache)
        engine = TTEngine(
            cache_dir=args.tt_cache,
            gguf_dir=args.model,
            tokenizer_path=args.tokenizer,
            max_concurrency=args.max_concurrency,
            chunked_prefill=args.chunked_prefill,
            trace_region_bytes=args.trace_region_mb << 20,
        )
        logger.info("tt engine ready: %d layers on the mesh", engine.config.num_layers)
        return engine

    raise SystemExit(f"unknown backend {args.backend!r}")


def main() -> None:
    p = argparse.ArgumentParser(prog="ttrunner_qwen38_flash_next.server")
    p.add_argument("--model", required=True, help="directory holding the GGUF shards")
    p.add_argument("--tokenizer", help="path to tokenizer.json")
    p.add_argument("--backend", default="reference", choices=("reference", "tt"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-concurrency", type=int, default=1,
                   help="reference: 1 (not re-entrant); tt: concurrent sequences")
    p.add_argument("--cache-bytes", type=int, default=24 << 30)
    p.add_argument("--tt-cache", help="converted .tensorbin weight cache (tt backend)")
    p.add_argument("--trace-region-mb", type=int, default=128,
                   help="tt backend: device trace region to reserve")
    p.add_argument("--chunked-prefill", action="store_true",
                   help="tt backend: use the 128-token chunked prefill path")
    p.add_argument("--log-level", default="info")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    app = create_app(None)

    @app.on_event("startup")
    async def _startup() -> None:
        app.state.engine = build_engine(args)

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
