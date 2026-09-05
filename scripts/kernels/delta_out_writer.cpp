// delta_out_writer.cpp -- `delta`, which the state update still needs, and `out`.
//
// Compile-time args: 0 TPH, 1.. TensorAccessorArgs for delta, out
// Runtime args: 0 delta_addr, 1 out_addr, 2 head_lo, 3 head_hi

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t TPH = get_compile_time_arg_val(0);
    constexpr uint32_t cb_delta = 8, cb_out = 9;

    const uint32_t d_addr = get_arg_val<uint32_t>(0);
    const uint32_t o_addr = get_arg_val<uint32_t>(1);
    const uint32_t head_lo = get_arg_val<uint32_t>(2);
    const uint32_t head_hi = get_arg_val<uint32_t>(3);

    constexpr auto d_ta = TensorAccessorArgs<1>();
    const auto d_acc = TensorAccessor(d_ta, d_addr);
    constexpr auto o_ta = TensorAccessorArgs<d_ta.next_compile_time_args_offset()>();
    const auto o_acc = TensorAccessor(o_ta, o_addr);

    for (uint32_t h = head_lo; h < head_hi; ++h) {
        for (uint32_t j = 0; j < TPH; ++j) {
            cb_wait_front(cb_delta, 1);
            noc_async_write_page(TPH * h + j, d_acc, get_read_ptr(cb_delta));
            noc_async_write_barrier();
            cb_pop_front(cb_delta, 1);
        }
        for (uint32_t j = 0; j < TPH; ++j) {
            cb_wait_front(cb_out, 1);
            noc_async_write_page(TPH * h + j, o_acc, get_read_ptr(cb_out));
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
    }
}
