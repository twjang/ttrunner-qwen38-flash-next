"""Lazy, slice-capable access to GGUF weights.

Two tensors make eager dequantisation impossible on any machine:

    per_layer_token_embd   (320_001_536, 160)  IQ4_NL   28.8 GB  ->  205 GB f32
    ffn_{gate,up,down}_exps (512, 640, 2560)   IQ3_S     0.7 GB/layer x 48

Both are only ever touched a few rows at a time (10 of 512 experts per token;
16 n-gram rows per token), so this store dequantises *row ranges* directly out
of the mmap. That works because every quant block is contained within a single
row for these tensors -- checked at slice time rather than assumed.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np
import torch

from ..gguf.quants import dequantize
from ..gguf.reader import TYPE_TRAITS, GGUFModel, TensorInfo


class WeightStore:
    """Dequantises GGUF tensors on demand, with an LRU cache for dense weights."""

    # Above this many requested rows, bypass the LRU and dequantise in bulk.
    BULK_ROW_THRESHOLD = 32

    def __init__(
        self,
        model: GGUFModel,
        cache_bytes: int = 8 << 30,
        row_cache_bytes: int = 24 << 30,
        dtype: torch.dtype = torch.float32,
        quant_sim: "callable | None" = None,
    ):
        self.model = model
        self.dtype = dtype
        # Optional name -> device-dtype hook used to reproduce the ttnn engine's
        # block-float numerics inside the CPU reference (see tt/blockfloat.py).
        self.quant_sim = quant_sim
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._cache_bytes = 0
        self._cache_limit = cache_bytes
        # Decoding re-selects largely the same experts step after step, so
        # caching individual dequantised rows turns the dominant cost of a
        # decode step into a lookup.
        self._rows: OrderedDict[tuple[str, int], torch.Tensor] = OrderedDict()
        self._rows_bytes = 0
        self._rows_limit = row_cache_bytes
        self.row_hits = 0
        self.row_misses = 0
        self._lock = threading.Lock()

    # -- helpers ---------------------------------------------------------

    def info(self, name: str) -> TensorInfo:
        try:
            return self.model.tensors[name]
        except KeyError:
            raise KeyError(f"no tensor named {name!r} in checkpoint") from None

    def has(self, name: str) -> bool:
        return name in self.model.tensors

    def _to_torch(self, flat: np.ndarray, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.from_numpy(flat).reshape(shape).to(self.dtype)

    # -- full tensors ----------------------------------------------------

    def get(self, name: str) -> torch.Tensor:
        """Whole tensor, dequantised and cached."""
        with self._lock:
            if name in self._cache:
                self._cache.move_to_end(name)
                return self._cache[name]

        info = self.info(name)
        flat = dequantize(self.model.raw(name), info.ggml_type, info.n_elements)
        if self.quant_sim is not None:
            flat = self.quant_sim(name, flat)
        tensor = self._to_torch(flat, info.torch_shape)

        nbytes = tensor.element_size() * tensor.nelement()
        with self._lock:
            self._cache[name] = tensor
            self._cache_bytes += nbytes
            while self._cache_bytes > self._cache_limit and len(self._cache) > 1:
                evicted, t = self._cache.popitem(last=False)
                if evicted == name:  # never evict what we just asked for
                    self._cache[name] = t
                    self._cache.move_to_end(name)
                    break
                self._cache_bytes -= t.element_size() * t.nelement()
        return tensor

    # -- row slices ------------------------------------------------------

    def _row_geometry(self, info: TensorInfo) -> tuple[int, int, int]:
        """(elements per row, bytes per row, rows) for first-dim slicing."""
        shape = info.torch_shape
        if len(shape) < 2:
            raise ValueError(f"{info.name}: cannot row-slice a 1-D tensor")
        row_elems = 1
        for d in shape[1:]:
            row_elems *= d
        block_size, type_size = TYPE_TRAITS[info.ggml_type]
        if row_elems % block_size:
            raise ValueError(
                f"{info.name}: row of {row_elems} elements straddles "
                f"{info.ggml_type.name} blocks of {block_size}; cannot row-slice"
            )
        return row_elems, row_elems // block_size * type_size, shape[0]

    def get_rows(self, name: str, indices: torch.Tensor | np.ndarray | list[int]) -> torch.Tensor:
        """Dequantise only the given first-dimension rows.

        Returns shape (len(indices), *tensor.shape[1:]).
        """
        info = self.info(name)
        row_elems, row_bytes, n_rows = self._row_geometry(info)
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        if idx.size and (idx.min() < 0 or idx.max() >= n_rows):
            raise IndexError(f"{name}: row index out of range [0, {n_rows})")

        raw = self.model.raw(name)
        tail = info.torch_shape[1:]

        # Two regimes, because they want opposite things:
        #
        # * Many rows (the PLE n-gram gather: batch x 16 heads, essentially all
        #   distinct and never reused) -- go fully vectorised. One fancy-index
        #   for the bytes, one `dequantize` for the block, one `index_select` to
        #   expand duplicates. No Python per row, no cache churn.
        # * Few rows (MoE experts: <= 10 per layer, heavily reused across decode
        #   steps) -- go through the LRU, where the hit rate is what matters.
        #
        # The per-row loop that served both made the single PLE layer cost 51 ms
        # at batch 64 -- six times a whole DeltaNet layer.
        if idx.size > self.BULK_ROW_THRESHOLD:
            unique, inverse = np.unique(idx, return_inverse=True)
            offsets = (unique[:, None] * row_bytes
                       + np.arange(row_bytes, dtype=np.int64)[None, :])
            gathered = np.frombuffer(raw, dtype=np.uint8)[offsets].reshape(-1)
            flat = dequantize(memoryview(gathered.tobytes()), info.ggml_type, unique.size * row_elems)
            block = self._to_torch(flat, (unique.size, *tail))
            return block.index_select(0, torch.from_numpy(inverse.astype(np.int64)))

        pieces: list[torch.Tensor] = []
        for r in idx:
            key = (name, int(r))
            with self._lock:
                hit = self._rows.get(key)
                if hit is not None:
                    self._rows.move_to_end(key)
                    self.row_hits += 1
            if hit is not None:
                pieces.append(hit)
                continue
            self.row_misses += 1
            start = int(r) * row_bytes
            row = self._to_torch(
                dequantize(raw[start : start + row_bytes], info.ggml_type, row_elems), tail
            )
            self._remember_row(key, row)
            pieces.append(row)
        return torch.stack(pieces, dim=0)

    def _remember_row(self, key: tuple[str, int], row: torch.Tensor) -> None:
        nbytes = row.element_size() * row.nelement()
        if nbytes > self._rows_limit:
            return
        with self._lock:
            self._rows[key] = row
            self._rows_bytes += nbytes
            while self._rows_bytes > self._rows_limit and self._rows:
                _, evicted = self._rows.popitem(last=False)
                self._rows_bytes -= evicted.element_size() * evicted.nelement()

    def get_row_range(self, name: str, start: int, stop: int) -> torch.Tensor:
        """Contiguous row range -- one dequant call instead of len(range)."""
        info = self.info(name)
        row_elems, row_bytes, n_rows = self._row_geometry(info)
        if not (0 <= start <= stop <= n_rows):
            raise IndexError(f"{name}: bad row range [{start}, {stop}) of {n_rows}")
        n = stop - start
        raw = self.model.raw(name)[start * row_bytes : stop * row_bytes]
        flat = dequantize(raw, info.ggml_type, n * row_elems)
        if self.quant_sim is not None:
            flat = self.quant_sim(name, flat)
        return self._to_torch(flat, (n, *info.torch_shape[1:]))

    # -- introspection ---------------------------------------------------

    def cache_stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._cache),
                "bytes": self._cache_bytes,
                "limit": self._cache_limit,
                "row_entries": len(self._rows),
                "row_bytes": self._rows_bytes,
                "row_hits": self.row_hits,
                "row_misses": self.row_misses,
            }

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()
            self._cache_bytes = 0
            self._rows.clear()
            self._rows_bytes = 0
