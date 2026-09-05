// mcast_probe.cpp -- does a NOC multicast work inside `ttnn.generic_op`?
//
// The gathering GEMV reads the activation row once **per core** -- 2.1 MB of the
// 6.9 the gate|up projection moves. `ttnn`'s own matmul multicasts it
// (`mcast_in0=True`); this kernel cannot, because a `generic_op` has to do its
// own. That is worth ~1.1 ms across the two projections, and this file is the
// smallest thing that answers whether the machinery works before any of it goes
// near the model.
//
// One core reads a page and broadcasts it to every core's landing buffer; each
// core then writes what it received to its own output page. If the multicast
// works, all N output pages equal input page 0.
//
// The sender is inside the destination rectangle, so the loopback form is used
// and `num_dests` counts every core including the sender.
//
// Compile-time args: 0 N_DEST, 1 PAGE, 2.. TensorAccessorArgs for src, out
// Runtime args: 0 src, 1 out, 2 is_sender, 3 mx0, 4 my0, 5 mx1, 6 my1,
//               7 sender_x, 8 sender_y, 9 my_page

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t N_DEST = get_compile_time_arg_val(0);
    constexpr uint32_t PAGE = get_compile_time_arg_val(1);
    constexpr uint32_t cb_a = 0;

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t is_sender = get_arg_val<uint32_t>(2);
    const uint32_t mx0 = get_arg_val<uint32_t>(3);
    const uint32_t my0 = get_arg_val<uint32_t>(4);
    const uint32_t mx1 = get_arg_val<uint32_t>(5);
    const uint32_t my1 = get_arg_val<uint32_t>(6);
    const uint32_t sx = get_arg_val<uint32_t>(7);
    const uint32_t sy = get_arg_val<uint32_t>(8);
    const uint32_t my_page = get_arg_val<uint32_t>(9);

    constexpr auto s_ta = TensorAccessorArgs<2>();
    const auto s_acc = TensorAccessor(s_ta, src_addr);
    constexpr auto o_ta = TensorAccessorArgs<s_ta.next_compile_time_args_offset()>();
    const auto o_acc = TensorAccessor(o_ta, out_addr);

    const uint32_t land = get_write_ptr(cb_a);
    volatile tt_l1_ptr uint32_t* ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    volatile tt_l1_ptr uint32_t* valid =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));

    if (is_sender) {
        noc_async_read_page(0, s_acc, land);
        noc_async_read_barrier();
        // Every other core signals once it has cleared its own flag.
        noc_semaphore_wait(ready, N_DEST - 1);
        noc_semaphore_set(ready, 0);

        const uint64_t dst = get_noc_multicast_addr(mx0, my0, mx1, my1, land);
        noc_async_write_multicast_loopback_src(land, dst, PAGE, N_DEST);
        noc_async_write_barrier();

        noc_semaphore_set(valid, 1);
        const uint64_t sdst =
            get_noc_multicast_addr(mx0, my0, mx1, my1, (uint32_t)get_semaphore(1));
        noc_semaphore_set_multicast_loopback_src(
            (uint32_t)get_semaphore(1), sdst, N_DEST);
    } else {
        noc_semaphore_set(valid, 0);
        const uint64_t s = get_noc_addr(sx, sy, (uint32_t)get_semaphore(0));
        noc_semaphore_inc(s, 1);
        noc_semaphore_wait(valid, 1);
    }

    noc_async_write_page(my_page, o_acc, land);
    noc_async_write_barrier();
}
