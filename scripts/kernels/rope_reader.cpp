// rope_reader.cpp -- feed a fused rotary embedding.
//
// `_apply_rope_dev` is eleven ttnn ops -- six slices, a negate, two concats, two
// multiplies and an add -- and it runs three dozen times a token on tensors of
// eight tiles or fewer. At 5.8 us an op whatever its shape (invariant 66) that
// is ~64 us a call and about 3 ms of a 49 ms step, to rotate 64 of 256 channels.
//
// The whole thing collapses because **rope_dim is 64 and half is 32, which is
// exactly one tile**. The rotation `[-second, first]` is therefore a swap of two
// whole tiles, not a shuffle inside one, and every output tile is either a fused
// pair of multiplies or a copy:
//
//   out tile 0 = x0 * cos0 - x1 * sin0
//   out tile 1 = x1 * cos1 + x0 * sin1
//   out tile j = x j                      (j >= 2, the un-rotated channels)
//
// Compile-time args:
//   0: NT        (tiles across a row of x: head_dim / 32)
//   1: NT_ROPE   (tiles across the rotated part: rope_dim / 32, must be 2)
//   2..: TensorAccessorArgs for x, cos, sin
//
// Runtime args: 0 x_addr, 1 cos_addr, 2 sin_addr, 3 work_lo, 4 work_hi

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t NT = get_compile_time_arg_val(0);
constexpr uint32_t NT_ROPE = get_compile_time_arg_val(1);

void kernel_main() {
    constexpr uint32_t cb_a = 0;      // x tile 0 of the pair, or the passthrough
    constexpr uint32_t cb_b = 1;      // x tile 1 of the pair
    constexpr uint32_t cb_cos = 2;
    constexpr uint32_t cb_sin = 3;

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t cos_addr = get_arg_val<uint32_t>(1);
    const uint32_t sin_addr = get_arg_val<uint32_t>(2);
    const uint32_t work_lo = get_arg_val<uint32_t>(3);
    const uint32_t work_hi = get_arg_val<uint32_t>(4);

    constexpr auto x_ta = TensorAccessorArgs<2>();
    const auto x_acc = TensorAccessor(x_ta, x_addr);
    constexpr auto c_ta = TensorAccessorArgs<x_ta.next_compile_time_args_offset()>();
    const auto c_acc = TensorAccessor(c_ta, cos_addr);
    constexpr auto s_ta = TensorAccessorArgs<c_ta.next_compile_time_args_offset()>();
    const auto s_acc = TensorAccessor(s_ta, sin_addr);

    for (uint32_t w = work_lo; w < work_hi; ++w) {
        const uint32_t r = w / NT;
        const uint32_t c = w - r * NT;

        if (c < NT_ROPE) {
            cb_reserve_back(cb_a, 1);
            noc_async_read_page(r * NT + 0, x_acc, get_write_ptr(cb_a));
            cb_reserve_back(cb_b, 1);
            noc_async_read_page(r * NT + 1, x_acc, get_write_ptr(cb_b));
            cb_reserve_back(cb_cos, 1);
            noc_async_read_page(r * NT_ROPE + c, c_acc, get_write_ptr(cb_cos));
            cb_reserve_back(cb_sin, 1);
            noc_async_read_page(r * NT_ROPE + c, s_acc, get_write_ptr(cb_sin));
            noc_async_read_barrier();
            cb_push_back(cb_a, 1);
            cb_push_back(cb_b, 1);
            cb_push_back(cb_cos, 1);
            cb_push_back(cb_sin, 1);
        } else {
            cb_reserve_back(cb_a, 1);
            noc_async_read_page(w, x_acc, get_write_ptr(cb_a));
            noc_async_read_barrier();
            cb_push_back(cb_a, 1);
        }
    }
}
