// delta_tail_writer.cpp -- the gated head output, in the flat [1, 1, 1, H*hd].
//
// Compile-time args: 0 TPH, 1.. TensorAccessorArgs for the output
// Runtime args: 0 out_addr, 1 head_lo, 2 head_hi

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t TPH = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = 4;

    const uint32_t head_lo = get_arg_val<uint32_t>(1);
    const uint32_t head_hi = get_arg_val<uint32_t>(2);

    constexpr auto o_ta = TensorAccessorArgs<1>();
    const auto o_acc = TensorAccessor(o_ta, get_arg_val<uint32_t>(0));

    for (uint32_t h = head_lo; h < head_hi; ++h) {
        for (uint32_t j = 0; j < TPH; ++j) {
            cb_wait_front(cb_out, 1);
            noc_async_write_page(TPH * h + j, o_acc, get_read_ptr(cb_out));
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
    }
}
