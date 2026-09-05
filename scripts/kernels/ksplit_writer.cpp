// ksplit_writer.cpp -- park this core's partial in its own slot.
//
// The output is [1, G, M, N] and this core owns (g, nt), so its tile is
// g * NT + nt. The host then sums over G with one `ttnn.sum`, which is what
// replaces a cross-core reduction.
//
// A core the plan has nothing for writes **nothing**. It would otherwise wait
// on a circular buffer its compute kernel never fills -- or, worse, take the
// garbage that kernel packed from an untouched destination register and put it
// in tile (0, 0), which every other idle core is also doing.
//
// Compile-time args: 0 NT, 1.. TensorAccessorArgs for the output.
// Per-core runtime args: 0 out_addr, 1 g, 2 nt, 3 active

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t NT = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = 16;

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t g = get_arg_val<uint32_t>(1);
    const uint32_t nt = get_arg_val<uint32_t>(2);
    const uint32_t active = get_arg_val<uint32_t>(3);

    constexpr auto o_ta = TensorAccessorArgs<1>();
    const auto o_acc = TensorAccessor(o_ta, out_addr);

    if (active == 0) {
        return;
    }

    cb_wait_front(cb_out, 1);
    noc_async_write_page(g * NT + nt, o_acc, get_read_ptr(cb_out));
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
}
