"""Minimal GGUF v3 reader.

Parses the header/metadata of a GGUF file and memory-maps the tensor data so
that shards totalling ~188 GB can be opened without reading them into RAM.
Only the subset of the format this project needs is implemented; anything
unexpected raises rather than being silently skipped.
"""

from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

GGUF_MAGIC = b"GGUF"


class ValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


_SCALAR_FMT = {
    ValueType.UINT8: "<B",
    ValueType.INT8: "<b",
    ValueType.UINT16: "<H",
    ValueType.INT16: "<h",
    ValueType.UINT32: "<I",
    ValueType.INT32: "<i",
    ValueType.FLOAT32: "<f",
    ValueType.BOOL: "<?",
    ValueType.UINT64: "<Q",
    ValueType.INT64: "<q",
    ValueType.FLOAT64: "<d",
}


class GGMLType(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30


# (block size in elements, bytes per block)
TYPE_TRAITS: dict[GGMLType, tuple[int, int]] = {
    GGMLType.F32: (1, 4),
    GGMLType.F16: (1, 2),
    GGMLType.BF16: (1, 2),
    GGMLType.F64: (1, 8),
    GGMLType.I8: (1, 1),
    GGMLType.I16: (1, 2),
    GGMLType.I32: (1, 4),
    GGMLType.I64: (1, 8),
    GGMLType.Q4_0: (32, 18),
    GGMLType.Q4_1: (32, 20),
    GGMLType.Q5_0: (32, 22),
    GGMLType.Q5_1: (32, 24),
    GGMLType.Q8_0: (32, 34),
    GGMLType.Q8_1: (32, 36),
    GGMLType.Q2_K: (256, 84),
    GGMLType.Q3_K: (256, 110),
    GGMLType.Q4_K: (256, 144),
    GGMLType.Q5_K: (256, 176),
    GGMLType.Q6_K: (256, 210),
    GGMLType.Q8_K: (256, 292),
    GGMLType.IQ4_NL: (32, 18),
    GGMLType.IQ4_XS: (256, 136),
    GGMLType.IQ1_S: (256, 50),
    GGMLType.IQ1_M: (256, 56),
    GGMLType.IQ2_XXS: (256, 66),
    GGMLType.IQ2_XS: (256, 74),
    GGMLType.IQ2_S: (256, 82),
    GGMLType.IQ3_XXS: (256, 98),
    GGMLType.IQ3_S: (256, 110),
}


def nbytes_for(ggml_type: GGMLType, n_elements: int) -> int:
    block_size, type_size = TYPE_TRAITS[ggml_type]
    if n_elements % block_size:
        raise ValueError(f"{n_elements} elements is not a multiple of {ggml_type.name} block {block_size}")
    return n_elements // block_size * type_size


@dataclass(slots=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]  # GGUF order (fastest-varying first)
    ggml_type: GGMLType
    offset: int  # relative to the file's tensor-data section
    file: Path
    data_start: int  # absolute offset of the tensor-data section

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def nbytes(self) -> int:
        return nbytes_for(self.ggml_type, self.n_elements)

    @property
    def torch_shape(self) -> tuple[int, ...]:
        """Shape in row-major (numpy/torch) order."""
        return tuple(reversed(self.shape))


class _Cursor:
    __slots__ = ("buf", "pos")

    def __init__(self, buf: memoryview, pos: int = 0) -> None:
        self.buf = buf
        self.pos = pos

    def read(self, fmt: str) -> Any:
        size = struct.calcsize(fmt)
        (value,) = struct.unpack_from(fmt, self.buf, self.pos)
        self.pos += size
        return value

    def read_string(self) -> str:
        length = self.read("<Q")
        raw = bytes(self.buf[self.pos : self.pos + length])
        self.pos += length
        return raw.decode("utf-8", errors="replace")

    def read_value(self, vtype: ValueType) -> Any:
        if vtype == ValueType.STRING:
            return self.read_string()
        if vtype == ValueType.ARRAY:
            elem_type = ValueType(self.read("<I"))
            count = self.read("<Q")
            if elem_type == ValueType.STRING:
                return [self.read_string() for _ in range(count)]
            fmt = _SCALAR_FMT[elem_type]
            size = struct.calcsize(fmt)
            # struct.unpack_from in a loop is far too slow for a 248k-entry
            # token list, so unpack the whole run at once.
            values = list(struct.unpack_from(f"<{count}{fmt[1:]}", self.buf, self.pos))
            self.pos += size * count
            return values
        return self.read(_SCALAR_FMT[vtype])


class GGUFFile:
    """A single .gguf file: metadata plus an mmap over its tensor data."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fd = open(self.path, "rb")
        self._mm = mmap.mmap(self._fd.fileno(), 0, access=mmap.ACCESS_READ)
        self.view = memoryview(self._mm)

        cur = _Cursor(self.view)
        magic = bytes(self.view[:4])
        if magic != GGUF_MAGIC:
            raise ValueError(f"{self.path.name}: not a GGUF file (magic {magic!r})")
        cur.pos = 4
        self.version = cur.read("<I")
        if self.version != 3:
            raise ValueError(f"{self.path.name}: unsupported GGUF version {self.version}")
        n_tensors = cur.read("<Q")
        n_kv = cur.read("<Q")

        self.metadata: dict[str, Any] = {}
        for _ in range(n_kv):
            key = cur.read_string()
            vtype = ValueType(cur.read("<I"))
            self.metadata[key] = cur.read_value(vtype)

        raw_tensors = []
        for _ in range(n_tensors):
            name = cur.read_string()
            n_dims = cur.read("<I")
            shape = tuple(cur.read("<Q") for _ in range(n_dims))
            ggml_type = GGMLType(cur.read("<I"))
            offset = cur.read("<Q")
            raw_tensors.append((name, shape, ggml_type, offset))

        alignment = self.metadata.get("general.alignment", 32)
        self.data_start = (cur.pos + alignment - 1) // alignment * alignment

        self.tensors: dict[str, TensorInfo] = {
            name: TensorInfo(name, shape, ggml_type, offset, self.path, self.data_start)
            for name, shape, ggml_type, offset in raw_tensors
        }

    def raw(self, info: TensorInfo) -> memoryview:
        start = self.data_start + info.offset
        return self.view[start : start + info.nbytes]

    def close(self) -> None:
        self.view.release()
        self._mm.close()
        self._fd.close()

    def __enter__(self) -> GGUFFile:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class GGUFModel:
    """A (possibly split) GGUF checkpoint presented as one tensor namespace."""

    def __init__(self, paths: list[str | Path]) -> None:
        if not paths:
            raise ValueError("no GGUF files given")
        self.files = [GGUFFile(p) for p in sorted(paths, key=lambda p: str(p))]
        self.metadata = self.files[0].metadata
        self.tensors: dict[str, TensorInfo] = {}
        self._owner: dict[str, GGUFFile] = {}
        for f in self.files:
            for name, info in f.tensors.items():
                if name in self.tensors:
                    raise ValueError(f"tensor {name!r} appears in two shards")
                self.tensors[name] = info
                self._owner[name] = f

    @classmethod
    def from_dir(cls, directory: str | Path, pattern: str = "*.gguf") -> GGUFModel:
        files = sorted(Path(directory).glob(pattern))
        if not files:
            raise FileNotFoundError(f"no files matching {pattern} in {directory}")
        return cls(list(files))

    def raw(self, name: str) -> memoryview:
        return self._owner[name].raw(self.tensors[name])

    def close(self) -> None:
        for f in self.files:
            f.close()

    def __enter__(self) -> GGUFModel:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
