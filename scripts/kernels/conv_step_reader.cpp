// conv_step_reader.cpp -- feed the fused 4-tap causal conv, and advance its ring.
//
// The decode conv is eleven wide ops a layer: four multiplies, three adds, three
// ring copies and a silu, every one of them on a [1, B, 1, C] tensor that is 144
// tiles at batch 1. That is 396 launches a token for arithmetic whose bytes are
// worth about nine microseconds a layer.
//
// This reader fetches one tile of each of the eight inputs -- the new column, the
// three history columns, and the four per-channel taps -- and then does the ring
// shift itself. The shift is safe here because every source is already in L1 by
// the time any destination is written, and because tile `t` of every tensor
// belongs to exactly one core, so no other core can be reading what this one
// overwrites.
//
// Compile-time args:
//   0: TILE_BYTES
//   1..: TensorAccessorArgs for x, s0, s1, s2, w0, w1, w2, w3
//        (one accessor, reused for all four taps, is not possible: they are four
//         separate tensors with four separate addresses)
//
// Runtime args: 0 x, 1 s0, 2 s1, 3 s2, 4 w0, 5 w1, 6 w2, 7 w3, 8 work_lo, 9 work_hi

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t work_lo = get_arg_val<uint32_t>(8);
    const uint32_t work_hi = get_arg_val<uint32_t>(9);

    constexpr auto x_ta = TensorAccessorArgs<1>();
    const auto x_acc = TensorAccessor(x_ta, get_arg_val<uint32_t>(0));
    constexpr auto s0_ta = TensorAccessorArgs<x_ta.next_compile_time_args_offset()>();
    const auto s0_acc = TensorAccessor(s0_ta, get_arg_val<uint32_t>(1));
    constexpr auto s1_ta = TensorAccessorArgs<s0_ta.next_compile_time_args_offset()>();
    const auto s1_acc = TensorAccessor(s1_ta, get_arg_val<uint32_t>(2));
    constexpr auto s2_ta = TensorAccessorArgs<s1_ta.next_compile_time_args_offset()>();
    const auto s2_acc = TensorAccessor(s2_ta, get_arg_val<uint32_t>(3));
    constexpr auto w0_ta = TensorAccessorArgs<s2_ta.next_compile_time_args_offset()>();
    const auto w0_acc = TensorAccessor(w0_ta, get_arg_val<uint32_t>(4));
    constexpr auto w1_ta = TensorAccessorArgs<w0_ta.next_compile_time_args_offset()>();
    const auto w1_acc = TensorAccessor(w1_ta, get_arg_val<uint32_t>(5));
    constexpr auto w2_ta = TensorAccessorArgs<w1_ta.next_compile_time_args_offset()>();
    const auto w2_acc = TensorAccessor(w2_ta, get_arg_val<uint32_t>(6));
    constexpr auto w3_ta = TensorAccessorArgs<w2_ta.next_compile_time_args_offset()>();
    const auto w3_acc = TensorAccessor(w3_ta, get_arg_val<uint32_t>(7));

    constexpr uint32_t cb_x = 0, cb_s0 = 1, cb_s1 = 2, cb_s2 = 3;
    constexpr uint32_t cb_w0 = 4, cb_w1 = 5, cb_w2 = 6, cb_w3 = 7;

    for (uint32_t w = work_lo; w < work_hi; ++w) {
        cb_reserve_back(cb_x, 1);
        cb_reserve_back(cb_s0, 1);
        cb_reserve_back(cb_s1, 1);
        cb_reserve_back(cb_s2, 1);
        cb_reserve_back(cb_w0, 1);
        cb_reserve_back(cb_w1, 1);
        cb_reserve_back(cb_w2, 1);
        cb_reserve_back(cb_w3, 1);

        const uint32_t x_l1 = get_write_ptr(cb_x);
        const uint32_t s0_l1 = get_write_ptr(cb_s0);
        const uint32_t s1_l1 = get_write_ptr(cb_s1);
        noc_async_read_page(w, x_acc, x_l1);
        noc_async_read_page(w, s0_acc, s0_l1);
        noc_async_read_page(w, s1_acc, s1_l1);
        noc_async_read_page(w, s2_acc, get_write_ptr(cb_s2));
        noc_async_read_page(w, w0_acc, get_write_ptr(cb_w0));
        noc_async_read_page(w, w1_acc, get_write_ptr(cb_w1));
        noc_async_read_page(w, w2_acc, get_write_ptr(cb_w2));
        noc_async_read_page(w, w3_acc, get_write_ptr(cb_w3));
        noc_async_read_barrier();

        // The ring, oldest first, from L1 copies of the values just read -- so
        // the three writes are independent of each other's order.
        noc_async_write_page(w, s2_acc, s1_l1);
        noc_async_write_page(w, s1_acc, s0_l1);
        noc_async_write_page(w, s0_acc, x_l1);
        // Before the push, not after: the next iteration reserves these same
        // slots, and a write still in flight would then be sourcing from a
        // buffer the reader has begun refilling.
        noc_async_write_barrier();

        cb_push_back(cb_x, 1);
        cb_push_back(cb_s0, 1);
        cb_push_back(cb_s1, 1);
        cb_push_back(cb_s2, 1);
        cb_push_back(cb_w0, 1);
        cb_push_back(cb_w1, 1);
        cb_push_back(cb_w2, 1);
        cb_push_back(cb_w3, 1);
    }
}
