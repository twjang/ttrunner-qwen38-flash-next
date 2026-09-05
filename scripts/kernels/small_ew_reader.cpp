// small_ew_reader.cpp -- stream tiles for a right-sized elementwise op.
//
// ttnn's elementwise ops take the whole grid whatever the tensor: a one-tile
// `ttnn.multiply` costs 5.78 us, which `dispatch_floor.py` shows is the price of
// a 110-core launch, where a one-core launch is 2.06. Half this model's 5764
// calls a step touch 64 tiles or fewer, so ~10 ms of the step is spent starting
// cores that have nothing to do.
//
// This is the reader for a kernel sized to its data instead.
//
// It also does a **tile-aligned slice**, because a slice along the last dim at
// tile boundaries is just a strided page copy and `ttnn.slice` is 648 of the
// step's small calls. `IN_BASE` is the first source tile column and `IN_NT` the
// source row stride; the output's own stride is `OUT_NT`.
//
// Compile-time args:
//   0: BINARY    (1: two inputs, 0: one)
//   1: TILE_BYTES
//   2: IN_BASE   (first source tile column)
//   3: IN_NT     (source tiles a row)
//   4: OUT_NT    (output tiles a row)
//   5..: TensorAccessorArgs for a, then b if BINARY
//
// Runtime args: 0 a_addr, 1 b_addr (0 if unary), 2 work_lo, 3 work_hi

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t BINARY = get_compile_time_arg_val(0);
    constexpr uint32_t IN_BASE = get_compile_time_arg_val(2);
    constexpr uint32_t IN_NT = get_compile_time_arg_val(3);
    constexpr uint32_t OUT_NT = get_compile_time_arg_val(4);
    constexpr uint32_t cb_a = 0;
    constexpr uint32_t cb_b = 1;

    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t work_lo = get_arg_val<uint32_t>(2);
    const uint32_t work_hi = get_arg_val<uint32_t>(3);

    constexpr auto a_ta = TensorAccessorArgs<5>();
    const auto a_acc = TensorAccessor(a_ta, a_addr);
    constexpr auto b_ta = TensorAccessorArgs<a_ta.next_compile_time_args_offset()>();
    const auto b_acc = TensorAccessor(b_ta, b_addr);

    for (uint32_t w = work_lo; w < work_hi; ++w) {
        // Same layout in and out unless this is a slice, in which case the
        // output's row w/OUT_NT starts at column IN_BASE of the source's.
        const uint32_t r = w / OUT_NT;
        const uint32_t src = (IN_NT == OUT_NT && IN_BASE == 0)
                                 ? w
                                 : r * IN_NT + IN_BASE + (w - r * OUT_NT);
        cb_reserve_back(cb_a, 1);
        noc_async_read_page(src, a_acc, get_write_ptr(cb_a));
        if (BINARY) {
            cb_reserve_back(cb_b, 1);
            noc_async_read_page(src, b_acc, get_write_ptr(cb_b));
        }
        noc_async_read_barrier();
        cb_push_back(cb_a, 1);
        if (BINARY) {
            cb_push_back(cb_b, 1);
        }
    }
}
