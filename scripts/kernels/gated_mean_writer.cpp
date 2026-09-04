// gated_mean_writer.cpp -- drain the gated-average result out.
//
// The output tile index is the work index itself: the output has Nt tiles a row
// where the input has 2Nt, so out(row, nt) = row * Nt + nt = w.
//
// Compile-time args: 0 TILE_BYTES, 1.. TensorAccessorArgs for the output.
// Runtime args: 0 out_addr, 1 work_lo, 2 work_hi

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = 2;

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t work_lo = get_arg_val<uint32_t>(1);
    const uint32_t work_hi = get_arg_val<uint32_t>(2);

    constexpr auto o_ta = TensorAccessorArgs<1>();
    const auto o_acc = TensorAccessor(o_ta, out_addr);

    for (uint32_t w = work_lo; w < work_hi; ++w) {
        cb_wait_front(cb_out, 1);
        noc_async_write_page(w, o_acc, get_read_ptr(cb_out));
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
