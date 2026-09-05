// gnorm1_reader.cpp -- the hyper-connection norm in ONE launch.
//
// It was two: one pass reducing each group to a scale, a second scaling every
// tile by it. Two launches because the second cannot start until the first has
// written the scale -- which, inside one launch, a semaphore says just as well.
//
// Each core owns a run of tiles from exactly one group and reads them **once**;
// they stay in L1 for both phases, so the stream crosses DRAM once instead of
// twice.
//
//   phase 1  every core: sum of squares over its own tiles -> one partial
//   phase 2  the group's gatherer: fold the partials, rsqrt, multicast the scale
//   phase 3  every core: x * scale * weight, and this device's own group
//
// The coordination lives here rather than in the writer because this is the
// kernel `gather_gemv_reader` already multicasts from, on the NOC the host's
// corner ordering is written for. A rectangle handed to the other NOC wants its
// corners the other way round, and getting that wrong hangs the card.
//
// A group owns a whole number of grid rows, so its cores are a rectangle and the
// gatherer sits inside it: the loopback form, with N_DEST counting every core
// including the sender.
//
// Compile-time args: 0 PAGE, 1 N_DEST, 2.. TensorAccessorArgs for x, w
// Runtime args: 0 x, 1 w, 2 tile_lo, 3 tile_len, 4 parts, 5 is_gatherer,
//               6 gx, 7 gy, 8 mx0, 9 my0, 10 mx1, 11 my1, 12 slot

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t PAGE = get_compile_time_arg_val(0);
    constexpr uint32_t N_DEST = get_compile_time_arg_val(1);
    constexpr uint32_t cb_x = 0, cb_w = 1, cb_part = 2, cb_fold = 3,
                       cb_scale = 4, cb_bcast = 8;

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t lo = get_arg_val<uint32_t>(2);
    const uint32_t len = get_arg_val<uint32_t>(3);
    const uint32_t parts = get_arg_val<uint32_t>(4);
    const uint32_t is_gatherer = get_arg_val<uint32_t>(5);
    const uint32_t gx = get_arg_val<uint32_t>(6);
    const uint32_t gy = get_arg_val<uint32_t>(7);
    const uint32_t mx0 = get_arg_val<uint32_t>(8);
    const uint32_t my0 = get_arg_val<uint32_t>(9);
    const uint32_t mx1 = get_arg_val<uint32_t>(10);
    const uint32_t my1 = get_arg_val<uint32_t>(11);
    const uint32_t slot = get_arg_val<uint32_t>(12);

    constexpr auto x_ta = TensorAccessorArgs<2>();
    const auto x_acc = TensorAccessor(x_ta, x_addr);
    constexpr auto w_ta = TensorAccessorArgs<x_ta.next_compile_time_args_offset()>();
    const auto w_acc = TensorAccessor(w_ta, w_addr);

    if (len == 0) {
        return;
    }
    cb_reserve_back(cb_x, len);
    cb_reserve_back(cb_w, len);
    const uint32_t xb = get_write_ptr(cb_x);
    const uint32_t wb = get_write_ptr(cb_w);
    for (uint32_t j = 0; j < len; ++j) {
        noc_async_read_page(lo + j, x_acc, xb + j * PAGE);
        noc_async_read_page(lo + j, w_acc, wb + j * PAGE);
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, len);
    cb_push_back(cb_w, len);

    volatile tt_l1_ptr uint32_t* ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    volatile tt_l1_ptr uint32_t* have =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));

    // this core's partial -> its slot in the gatherer's fold buffer
    cb_wait_front(cb_part, 1);
    noc_async_write(get_read_ptr(cb_part),
                    get_noc_addr(gx, gy, get_write_ptr(cb_fold) + slot * PAGE), PAGE);
    noc_async_write_barrier();
    cb_pop_front(cb_part, 1);

    if (is_gatherer) {
        noc_semaphore_wait(ready, parts - 1);
        noc_semaphore_set(ready, 0);
        cb_reserve_back(cb_fold, parts);
        cb_push_back(cb_fold, parts);          // the tiles are already in place

        // The finished scale, straight into everyone's cb_scale. That buffer has
        // one page, so its address is the same on every core and stable across
        // the push -- which is what makes the destination computable here.
        cb_wait_front(cb_bcast, 1);
        const uint64_t dst =
            get_noc_multicast_addr(mx0, my0, mx1, my1, get_write_ptr(cb_scale));
        noc_async_write_multicast_loopback_src(get_read_ptr(cb_bcast), dst,
                                               PAGE, N_DEST);
        noc_async_write_barrier();
        cb_pop_front(cb_bcast, 1);

        noc_semaphore_set(have, 1);
        const uint64_t sdst =
            get_noc_multicast_addr(mx0, my0, mx1, my1, (uint32_t)get_semaphore(1));
        noc_semaphore_set_multicast_loopback_src(
            (uint32_t)get_semaphore(1), sdst, N_DEST);
    } else {
        noc_semaphore_set(have, 0);
        noc_semaphore_inc(get_noc_addr(gx, gy, (uint32_t)get_semaphore(0)), 1);
        noc_semaphore_wait(have, 1);
        cb_reserve_back(cb_scale, 1);
        cb_push_back(cb_scale, 1);             // the multicast landed there
    }
}
