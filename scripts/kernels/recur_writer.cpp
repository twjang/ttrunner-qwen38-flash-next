// recur_writer.cpp -- this core's output tile and its column of the new state.
//
// The state is written **in place**, back to the pages the reader took it from:
// `decode_step` keeps the recurrent state at a fixed device address across steps
// so the 48-layer graph can be captured as one trace, and this kernel inherits
// that contract. One write of DKT tiles a core, against the five passes the
// unfused chain makes.
//
// Compile-time args: 0 DKT, 1 DVT, 2 SPAGE, 3 OPAGE,
//                    4.. TensorAccessorArgs for out, state
// Runtime args: 0 out, 1 state, 2 head, 3 j, 4 active

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t DKT = get_compile_time_arg_val(0);
    constexpr uint32_t DVT = get_compile_time_arg_val(1);
    constexpr uint32_t SPAGE = get_compile_time_arg_val(2);
    constexpr uint32_t OPAGE = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = 16, cb_snew = 17;

    const uint32_t head = get_arg_val<uint32_t>(2);
    const uint32_t j = get_arg_val<uint32_t>(3);
    const uint32_t active = get_arg_val<uint32_t>(4);
    if (active == 0) {
        return;
    }

    constexpr auto o_ta = TensorAccessorArgs<4>();
    const auto o_acc = TensorAccessor(o_ta, get_arg_val<uint32_t>(0));
    constexpr auto s_ta = TensorAccessorArgs<o_ta.next_compile_time_args_offset()>();
    const auto s_acc = TensorAccessor(s_ta, get_arg_val<uint32_t>(1));

    cb_wait_front(cb_out, 1);
    noc_async_write_page(head * DVT + j, o_acc, get_read_ptr(cb_out));
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);

    cb_wait_front(cb_snew, DKT);
    const uint32_t sb = get_read_ptr(cb_snew);
    for (uint32_t i = 0; i < DKT; ++i) {
        noc_async_write_page(head * DKT * DVT + i * DVT + j, s_acc, sb + i * SPAGE);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_snew, DKT);
}
