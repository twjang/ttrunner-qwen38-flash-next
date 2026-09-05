// outer_add_writer.cpp -- the new state, straight into the state buffer.
//
// Compile-time args: 0 PAGE, 1.. accessor for state
// Runtime args: 0 state, 1 tile_lo, 2 tile_len

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = 16;
    const uint32_t s_addr = get_arg_val<uint32_t>(0);
    const uint32_t lo = get_arg_val<uint32_t>(1);
    const uint32_t len = get_arg_val<uint32_t>(2);
    constexpr auto s_ta = TensorAccessorArgs<1>();
    const auto s_acc = TensorAccessor(s_ta, s_addr);
    if (len == 0) {
        return;
    }
    for (uint32_t j = 0; j < len; ++j) {
        cb_wait_front(cb_out, 1);
        noc_async_write_page(lo + j, s_acc, get_read_ptr(cb_out));
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
