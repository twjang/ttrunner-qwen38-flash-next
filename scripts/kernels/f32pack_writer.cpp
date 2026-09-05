// f32pack_writer.cpp -- drain the copies.
// Compile-time args: 0 PAGE, 1.. accessor for out
// Runtime args: 0 in, 1 out, 2 lo, 3 len
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = 16;
    const uint32_t o_addr = get_arg_val<uint32_t>(1);
    const uint32_t lo = get_arg_val<uint32_t>(2);
    const uint32_t len = get_arg_val<uint32_t>(3);
    constexpr auto o_ta = TensorAccessorArgs<1>();
    const auto o_acc = TensorAccessor(o_ta, o_addr);
    if (len == 0) return;
    for (uint32_t j = 0; j < len; ++j) {
        cb_wait_front(cb_out, 1);
        noc_async_write_page(lo + j, o_acc, get_read_ptr(cb_out));
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
