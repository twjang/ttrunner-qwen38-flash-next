// swiglu_reader.cpp -- feed the fused SwiGLU compute kernel.
//
// `expert_ffn` computes gate|up as one fused matmul into [1, E, M, 2N] and then
// spends four ttnn ops turning it into `silu(gate) * up`: two slices, a silu and
// a multiply. That chain measures 142 us a layer, 6.83 ms a token, and the
// reason is bytes rather than op count -- it moves about 61 MB a layer because
// every one of the four ops reads and writes the whole E=128 tensor. One fused
// pass moves 16.7 MB: read both halves once, write the result once.
//
// Tile addressing, for an interleaved TILE_LAYOUT [1, E, M, 2N]:
//   row = e * Mt + mt,   Nt = N / 32
//   gate tile = row * 2Nt + nt
//   up   tile = row * 2Nt + Nt + nt
//   out  tile = row * Nt + nt = w        <- the work index itself
//
// Compile-time args:
//   0: NT_OUT      (Nt: output tiles per row)
//   1: TILE_BYTES  (accessor's aligned page size)
//   2..: TensorAccessorArgs for the [1, E, M, 2N] input
//
// Runtime args: 0 in_addr, 1 work_lo, 2 work_hi

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t NT_OUT = get_compile_time_arg_val(0);
    constexpr uint32_t TILE_BYTES = get_compile_time_arg_val(1);
    constexpr uint32_t cb_gate = 0;
    constexpr uint32_t cb_up = 1;

    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t work_lo = get_arg_val<uint32_t>(1);
    const uint32_t work_hi = get_arg_val<uint32_t>(2);

    constexpr auto in_ta = TensorAccessorArgs<2>();
    const auto in_acc = TensorAccessor(in_ta, in_addr);

    for (uint32_t w = work_lo; w < work_hi; ++w) {
        const uint32_t row = w / NT_OUT;
        const uint32_t nt = w - row * NT_OUT;
        const uint32_t base = row * (2u * NT_OUT) + nt;

        cb_reserve_back(cb_gate, 1);
        noc_async_read_page(base, in_acc, get_write_ptr(cb_gate));
        cb_reserve_back(cb_up, 1);
        noc_async_read_page(base + NT_OUT, in_acc, get_write_ptr(cb_up));
        noc_async_read_barrier();
        cb_push_back(cb_gate, 1);
        cb_push_back(cb_up, 1);
    }
}
