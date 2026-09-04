// gated_mean_reader.cpp -- feed the fused hyper-connection gate-and-average.
//
// `gated_residual_mix` ends with `multiply(mix, normed)` over a 10240-wide pair
// and then averages the four hc streams, which was nine ttnn ops: the multiply,
// four tile-aligned slices, three adds and a scale. Each of those pays ~5.5 us
// of fixed per-op cost whatever it touches (invariant 42), and between them they
// round-trip ~11 MB a call.
//
// Fused, one output tile needs eight input tiles and writes one:
//
//   out[row, nt] = (1/hc) * sum_h  mix[row, h*NT_OUT + nt] * normed[row, ...]
//
// The flattened layout is stream-major, so stream h is tile columns
// [h*NT_OUT, (h+1)*NT_OUT) of the 10240-wide row -- the same fact the slices
// were relying on.
//
// Compile-time args:
//   0: NT_OUT   (hidden/32: output tiles per row)
//   1: HC       (hc_count)
//   2: TILE_BYTES
//   3..: TensorAccessorArgs for `mix`, then for `normed`
//
// Runtime args: 0 mix_addr, 1 normed_addr, 2 work_lo, 3 work_hi

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t NT_OUT = get_compile_time_arg_val(0);
    constexpr uint32_t HC = get_compile_time_arg_val(1);
    constexpr uint32_t cb_a = 0;
    constexpr uint32_t cb_b = 1;

    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t work_lo = get_arg_val<uint32_t>(2);
    const uint32_t work_hi = get_arg_val<uint32_t>(3);

    constexpr auto a_ta = TensorAccessorArgs<3>();
    const auto a_acc = TensorAccessor(a_ta, a_addr);
    constexpr auto b_ta = TensorAccessorArgs<a_ta.next_compile_time_args_offset()>();
    const auto b_acc = TensorAccessor(b_ta, b_addr);

    for (uint32_t w = work_lo; w < work_hi; ++w) {
        const uint32_t row = w / NT_OUT;
        const uint32_t nt = w - row * NT_OUT;
        const uint32_t base = row * (HC * NT_OUT) + nt;

        cb_reserve_back(cb_a, HC);
        cb_reserve_back(cb_b, HC);
        const uint32_t a_l1 = get_write_ptr(cb_a);
        const uint32_t b_l1 = get_write_ptr(cb_b);
        for (uint32_t h = 0; h < HC; ++h) {
            noc_async_read_page(base + h * NT_OUT, a_acc, a_l1 + h * get_tile_size(cb_a));
            noc_async_read_page(base + h * NT_OUT, b_acc, b_l1 + h * get_tile_size(cb_b));
        }
        noc_async_read_barrier();
        cb_push_back(cb_a, HC);
        cb_push_back(cb_b, HC);
    }
}
