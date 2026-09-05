// gather_gemv_writer.cpp -- one output tile per column this core owns.
//
// Compile-time args: TensorAccessorArgs for the output
// Runtime args: 0 out_addr, 1 col_lo, 2 col_hi

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = 16;
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t col_lo = get_arg_val<uint32_t>(1);
    const uint32_t col_hi = get_arg_val<uint32_t>(2);

    constexpr auto o_ta = TensorAccessorArgs<0>();
    const auto o_acc = TensorAccessor(o_ta, out_addr);

    for (uint32_t c = col_lo; c < col_hi; ++c) {
        cb_wait_front(cb_out, 1);
        noc_async_write_page(c, o_acc, get_read_ptr(cb_out));
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
