// fabreduce_reader.cpp -- the accumulator's reader for a two-chip fabric reduce.
//
// Waits for the sender's fused atomic-inc, then streams this chip's own partial
// and the received one into two circular buffers for the compute kernel to add.
// The wait is here rather than in the writer because this is the kernel that
// must not read `scratch` before the payload has landed.
//
// Compile-time args: 0 ROLE, 1 BYTES, 2 SEM_ID, 3 NTILES,
//                    4.. TensorAccessorArgs for src, then scratch
// Runtime args: 0 src_addr, 1 scratch_addr, 2 expected

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t ROLE = get_compile_time_arg_val(0);
    constexpr uint32_t BYTES = get_compile_time_arg_val(1);
    constexpr uint32_t SEM_ID = get_compile_time_arg_val(2);
    constexpr uint32_t NT = get_compile_time_arg_val(3);
    constexpr uint32_t cb_mine = 0, cb_theirs = 1;

    if constexpr (ROLE != 2) {
        return;
    }
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t scr_addr = get_arg_val<uint32_t>(1);
    const uint32_t expected = get_arg_val<uint32_t>(2);
    const uint32_t page = BYTES / NT;

    volatile tt_l1_ptr uint32_t* sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_ID));
    noc_semaphore_wait(sem, expected);
    noc_semaphore_set(sem, 0);

    constexpr auto s_ta = TensorAccessorArgs<4>();
    const auto s_acc = TensorAccessor(s_ta, src_addr);
    constexpr auto t_ta = TensorAccessorArgs<s_ta.next_compile_time_args_offset()>();
    const auto t_acc = TensorAccessor(t_ta, scr_addr);

    for (uint32_t i = 0; i < NT; ++i) {
        cb_reserve_back(cb_mine, 1);
        noc_async_read_page(i, s_acc, get_write_ptr(cb_mine));
        cb_reserve_back(cb_theirs, 1);
        noc_async_read_page(i, t_acc, get_write_ptr(cb_theirs));
        noc_async_read_barrier();
        cb_push_back(cb_mine, 1);
        cb_push_back(cb_theirs, 1);
    }
}
