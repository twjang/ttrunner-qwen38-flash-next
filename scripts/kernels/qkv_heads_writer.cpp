// qkv_heads_writer.cpp -- the normalised q and k, one head at a time.
//
// The compute kernel pushes TPH q tiles then TPH k tiles for each head, so the
// writer alternates between the two destinations. `v` never comes through here:
// it needs no arithmetic, so the reader copies it straight across.
//
// Compile-time args: 0 TPH, 1.. TensorAccessorArgs for q_out, then k_out
// Runtime args: 0 q_addr, 1 k_addr, 2 head_lo, 3 head_hi

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t TPH = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = 3;

    const uint32_t head_lo = get_arg_val<uint32_t>(2);
    const uint32_t head_hi = get_arg_val<uint32_t>(3);

    constexpr auto q_ta = TensorAccessorArgs<1>();
    const auto q_acc = TensorAccessor(q_ta, get_arg_val<uint32_t>(0));
    constexpr auto k_ta = TensorAccessorArgs<q_ta.next_compile_time_args_offset()>();
    const auto k_acc = TensorAccessor(k_ta, get_arg_val<uint32_t>(1));

    for (uint32_t h = head_lo; h < head_hi; ++h) {
        for (uint32_t j = 0; j < TPH; ++j) {
            cb_wait_front(cb_out, 1);
            noc_async_write_page(TPH * h + j, q_acc, get_read_ptr(cb_out));
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
        for (uint32_t j = 0; j < TPH; ++j) {
            cb_wait_front(cb_out, 1);
            noc_async_write_page(TPH * h + j, k_acc, get_read_ptr(cb_out));
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
    }
}
