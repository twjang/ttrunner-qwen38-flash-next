// fabreduce_reader.cpp -- both roles' reader for a two-chip fabric reduce.
//
// sender:      stream this chip's partial DRAM -> `cb_mine`, so the writer has
//              an **L1** address to send from. Handoff 45.41: passing a DRAM
//              `buffer_address()` to the fabric send is what made the first
//              attempt deliver the wrong bytes, and what `fabric_probe.py` has
//              been doing all along.
// accumulator: wait for the sender's fused atomic-inc -- at which point the
//              payload is already sitting in `cb_theirs`' L1, written there by
//              the packet -- then stream its own partial into `cb_mine` and
//              push both for the compute kernel.
//
// The sender aims at `get_write_ptr(cb_theirs)` **on its own core**: every chip
// runs the same program with the same CB list, so the allocator gives that CB
// the same L1 address everywhere, and no address has to cross from Python.
//
// Compile-time args: 0 ROLE, 1 NTILES, 2 SEM_ID, 3.. TensorAccessorArgs for src
// Runtime args: 0 src_addr, 1 expected

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t ROLE = get_compile_time_arg_val(0);
    constexpr uint32_t NT = get_compile_time_arg_val(1);
    constexpr uint32_t SEM_ID = get_compile_time_arg_val(2);
    constexpr uint32_t cb_mine = 0, cb_theirs = 1;

    if constexpr (ROLE == 0) {
        return;
    }
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    constexpr auto s_ta = TensorAccessorArgs<3>();
    const auto s_acc = TensorAccessor(s_ta, src_addr);

    if constexpr (ROLE == 2) {
        const uint32_t expected = get_arg_val<uint32_t>(1);
        volatile tt_l1_ptr uint32_t* sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_ID));
        noc_semaphore_wait(sem, expected);
        noc_semaphore_set(sem, 0);
        // The packet wrote straight into cb_theirs' L1; only the bookkeeping is
        // left, so the compute kernel's `cb_wait_front` is satisfied.
        cb_reserve_back(cb_theirs, NT);
        cb_push_back(cb_theirs, NT);
    }

    cb_reserve_back(cb_mine, NT);
    const uint32_t base = get_write_ptr(cb_mine);
    const uint32_t page = get_tile_size(cb_mine);
    for (uint32_t i = 0; i < NT; ++i) {
        noc_async_read_page(i, s_acc, base + i * page);
    }
    noc_async_read_barrier();
    cb_push_back(cb_mine, NT);
}
