// ksgemv_writer.cpp -- every group's partial to the gatherer, then the output.
//
// The gatherer for output tile n is the core that owns n in group 0. Every other
// group's owner of n writes its partial straight into the gatherer's fold buffer
// at its own slot -- the same circular buffer at the same L1 offset on every
// core -- and signals. The gatherer waits for the count and hands the whole
// buffer to its own compute, which adds them and packs the output tile.
//
// That is the `ttnn.sum` the first k-split needed, minus the launch and minus
// reading G x N tiles back out of DRAM to add G numbers a column.
//
// Compile-time args: 0 PART_PAGE, 1 FOLD, 2 SEM_ID, 3 NT,
//                    4.. TensorAccessorArgs for out
//
// FOLD = 0 keeps the partials apart: this core writes tile `g * NT + n` of a
// [1, G, 32, N] output and `fused_group_sum` adds the G of them in its own
// launch. That is a launch and a DRAM round trip the in-kernel fold saves, and
// it is the form that is known to work.
// Runtime args: 0 out, 1 n, 2 active, 3 is_gatherer, 4 G, 5 slot, 6 gx, 7 gy

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t PART_PAGE = get_compile_time_arg_val(0);
    constexpr uint32_t FOLD = get_compile_time_arg_val(1);
    constexpr uint32_t SEM_ID = get_compile_time_arg_val(2);
    constexpr uint32_t NT = get_compile_time_arg_val(3);
    constexpr uint32_t cb_part = 2, cb_fold = 3, cb_out = 16;

    const uint32_t o_addr = get_arg_val<uint32_t>(0);
    const uint32_t n = get_arg_val<uint32_t>(1);
    const uint32_t active = get_arg_val<uint32_t>(2);
    const uint32_t is_gatherer = get_arg_val<uint32_t>(3);
    const uint32_t g_count = get_arg_val<uint32_t>(4);
    const uint32_t slot = get_arg_val<uint32_t>(5);
    const uint32_t gx = get_arg_val<uint32_t>(6);
    const uint32_t gy = get_arg_val<uint32_t>(7);

    constexpr auto o_ta = TensorAccessorArgs<4>();
    const auto o_acc = TensorAccessor(o_ta, o_addr);

    if (active == 0) {
        return;
    }
    if constexpr (!FOLD) {
        cb_wait_front(cb_out, 1);
        noc_async_write_page(slot * NT + n, o_acc, get_read_ptr(cb_out));
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
        return;
    }
    cb_wait_front(cb_part, 1);
    noc_async_write(get_read_ptr(cb_part),
                    get_noc_addr(gx, gy, get_write_ptr(cb_fold) + slot * PART_PAGE),
                    PART_PAGE);
    noc_async_write_barrier();
    cb_pop_front(cb_part, 1);

    volatile tt_l1_ptr uint32_t* sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_ID));
    if (!is_gatherer) {
        noc_semaphore_inc(get_noc_addr(gx, gy, (uint32_t)get_semaphore(SEM_ID)), 1);
        // The increment is a **posted** atomic and this is the only place in
        // this project where one is followed by nothing at all -- every other
        // `noc_semaphore_inc` here is followed by a `noc_semaphore_wait`, which
        // flushes it. The gatherer spins on exact equality, so one increment
        // that never lands hangs it for ever.
        noc_async_atomic_barrier();
        return;
    }
    noc_semaphore_wait(sem, g_count - 1);
    noc_semaphore_set(sem, 0);
    cb_reserve_back(cb_fold, g_count);
    cb_push_back(cb_fold, g_count);            // the tiles are already in place

    cb_wait_front(cb_out, 1);
    noc_async_write_page(n, o_acc, get_read_ptr(cb_out));
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
}
