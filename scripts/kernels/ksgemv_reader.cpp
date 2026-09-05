// ksgemv_reader.cpp -- a GEMV whose reduction is split across cores, fed once.
//
// `ttnn.linear` at M = 1 gives every output tile to one core, so a narrow output
// leaves most of the grid idle: `[1,2560] x [2560,352]` is eleven tiles and runs
// at **19.6 %** of bandwidth, 12.58 us against a 2.47 roofline, ninety-six times
// a token. Splitting the *reduction* fills the grid.
//
// The first k-split in this project (`ksplit_*`) lost anyway, for two reasons
// this kernel fixes:
//
//   * it folded the partials with a separate `ttnn.sum`, a whole launch and a
//     G x N read to add G numbers a column. The fold is in the kernel now.
//   * every core in a k-group read the same activation tiles from DRAM. At
//     M = 1 an activation tile is thirty-two rows of padding around one real
//     one, so that duplication is *larger than the weight*: 1.80 MB against
//     0.96 for [2560, 352]. The group's head reads them once and multicasts.
//
// A group is a rectangle -- either `nt` consecutive cores of one row when `nt`
// divides the grid width, or whole rows -- so the head sits inside it and the
// loopback form applies, N_DEST counting every core in the rectangle. Cores in
// the rectangle past `nt` have no output tile but still take part in the
// handshake; a core that is counted and does not signal hangs the group.
//
// Compile-time args: 0 NT, 1 A_PAGE, 2 W_PAGE, 3 N_DEST, 4.. accessors a, w
// Runtime args: 0 a, 1 w, 2 kt_lo, 3 klen, 4 n, 5 active, 6 is_head,
//               7 mx0, 8 my0, 9 mx1, 10 my1, 11 hx, 12 hy

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t NT = get_compile_time_arg_val(0);
    constexpr uint32_t A_PAGE = get_compile_time_arg_val(1);
    constexpr uint32_t W_PAGE = get_compile_time_arg_val(2);
    constexpr uint32_t N_DEST = get_compile_time_arg_val(3);
    constexpr uint32_t cb_a = 0, cb_w = 1;

    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t kt_lo = get_arg_val<uint32_t>(2);
    const uint32_t klen = get_arg_val<uint32_t>(3);
    const uint32_t n = get_arg_val<uint32_t>(4);
    const uint32_t active = get_arg_val<uint32_t>(5);
    const uint32_t is_head = get_arg_val<uint32_t>(6);
    const uint32_t mx0 = get_arg_val<uint32_t>(7);
    const uint32_t my0 = get_arg_val<uint32_t>(8);
    const uint32_t mx1 = get_arg_val<uint32_t>(9);
    const uint32_t my1 = get_arg_val<uint32_t>(10);
    const uint32_t hx = get_arg_val<uint32_t>(11);
    const uint32_t hy = get_arg_val<uint32_t>(12);
    // 1 for a core inside some group's rectangle, 0 for one this plan does not
    // use at all. **Not** the same as `active`, which is 0 for a core that is in
    // the rectangle for the handshake only and must still take part in it.
    //
    // Without this a core outside every group ran with all-zero runtime args, so
    // `is_head` was 0, it took the non-head path below, and it
    // `noc_semaphore_inc`d `get_noc_addr(0, 0, ...)` -- the semaphore of
    // whichever core sits at (0, 0), which is group 0's head. That head then
    // reached its count before its real members had armed and multicast early,
    // leaving a member waiting for ever; the idle core then hung on its own
    // `valid` too. Invisible whenever the plan covers all 110 cores, which is
    // every configuration this project ships. Handoff 45.10 and invariant 130.
    const uint32_t in_group = get_arg_val<uint32_t>(13);
    if (in_group == 0) {
        return;
    }

    constexpr auto a_ta = TensorAccessorArgs<4>();
    const auto a_acc = TensorAccessor(a_ta, a_addr);
    constexpr auto w_ta = TensorAccessorArgs<a_ta.next_compile_time_args_offset()>();
    const auto w_acc = TensorAccessor(w_ta, w_addr);

    if (klen == 0) {
        return;                        // a core outside every group's rectangle
    }
    cb_reserve_back(cb_a, klen);
    const uint32_t ab = get_write_ptr(cb_a);

    // This core's column of the weight, all of it against one barrier: issued a
    // tile at a time with a barrier each, every core pays a serialised DRAM
    // round trip per tile and the split loses to the matmul it replaces.
    if (active) {
        cb_reserve_back(cb_w, klen);
        const uint32_t wb = get_write_ptr(cb_w);
        for (uint32_t i = 0; i < klen; ++i) {
            noc_async_read_page((kt_lo + i) * NT + n, w_acc, wb + i * W_PAGE);
        }
    }

    if constexpr (N_DEST == 1) {
        for (uint32_t i = 0; i < klen; ++i) {
            noc_async_read_page(kt_lo + i, a_acc, ab + i * A_PAGE);
        }
        noc_async_read_barrier();
    } else {
        volatile tt_l1_ptr uint32_t* ready =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
        volatile tt_l1_ptr uint32_t* valid =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
        if (is_head) {
            for (uint32_t i = 0; i < klen; ++i) {
                noc_async_read_page(kt_lo + i, a_acc, ab + i * A_PAGE);
            }
            noc_async_read_barrier();
            noc_semaphore_wait(ready, N_DEST - 1);
            noc_semaphore_set(ready, 0);
            noc_async_write_multicast_loopback_src(
                ab, get_noc_multicast_addr(mx0, my0, mx1, my1, ab),
                klen * A_PAGE, N_DEST);
            noc_async_write_barrier();
            noc_semaphore_set(valid, 1);
            noc_semaphore_set_multicast_loopback_src(
                (uint32_t)get_semaphore(1),
                get_noc_multicast_addr(mx0, my0, mx1, my1, (uint32_t)get_semaphore(1)),
                N_DEST);
        } else {
            noc_semaphore_set(valid, 0);
            noc_semaphore_inc(get_noc_addr(hx, hy, (uint32_t)get_semaphore(0)), 1);
            noc_semaphore_wait(valid, 1);
            noc_async_read_barrier();          // this core's weight column
        }
    }
    if (!active) {
        return;                        // in the rectangle for the handshake only
    }
    cb_push_back(cb_a, klen);
    cb_push_back(cb_w, klen);
}
