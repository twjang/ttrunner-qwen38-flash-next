// group_scale_reader.cpp -- x, its weight and its group's scale, per tile.
//
// The scale is one tile per group and consecutive work items mostly share a
// group, so it is read only when the group changes -- and its circular buffer
// has **one page**, so the cached write pointer is stable. Two pages would
// alternate the address, the cache would never hit, and the read would happen on
// every tile (invariant 84, which cost the fused reinject a millisecond).
//
// Compile-time args: 0 NT_G, 1 PAGE, 2.. TensorAccessorArgs for x, w, scale
// Runtime args: 0 x_addr, 1 w_addr, 2 s_addr, 3 lo, 4 hi

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t NT_G = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x = 0, cb_w = 1, cb_s = 2;

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t s_addr = get_arg_val<uint32_t>(2);
    const uint32_t lo = get_arg_val<uint32_t>(3);
    const uint32_t hi = get_arg_val<uint32_t>(4);

    constexpr auto x_ta = TensorAccessorArgs<2>();
    const auto x_acc = TensorAccessor(x_ta, x_addr);
    constexpr auto w_ta = TensorAccessorArgs<x_ta.next_compile_time_args_offset()>();
    const auto w_acc = TensorAccessor(w_ta, w_addr);
    constexpr auto s_ta = TensorAccessorArgs<w_ta.next_compile_time_args_offset()>();
    const auto s_acc = TensorAccessor(s_ta, s_addr);

    uint32_t have = 0xffffffffu;
    for (uint32_t c = lo; c < hi; ++c) {
        const uint32_t g = c / NT_G;
        cb_reserve_back(cb_s, 1);
        if (g != have) {
            noc_async_read_page(g, s_acc, get_write_ptr(cb_s));
            noc_async_read_barrier();
            have = g;
        }
        cb_push_back(cb_s, 1);

        cb_reserve_back(cb_x, 1);
        noc_async_read_page(c, x_acc, get_write_ptr(cb_x));
        cb_reserve_back(cb_w, 1);
        noc_async_read_page(c, w_acc, get_write_ptr(cb_w));
        noc_async_read_barrier();
        cb_push_back(cb_x, 1);
        cb_push_back(cb_w, 1);
    }
}
