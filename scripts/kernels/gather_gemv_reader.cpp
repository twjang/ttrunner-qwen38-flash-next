// gather_gemv_reader.cpp -- a GEMV whose weight tiles are chosen by an index
// tensor, so the expert gather never happens.
//
// Today the wide expert path copies the selected experts' weights into a compact
// tensor and then multiplies against it. The copy is **not** the cheap half:
//
//     gather gate|up  40.11 us      matmul gate|up  20.80 us
//     gather down     34.46 us      matmul down     15.68 us
//
// -- 3.58 ms a token to move weights that the matmul then reads again, so every
// selected expert's weights cross DRAM three times instead of once. Reading them
// through the index in the matmul itself is the whole saving.
//
// Each core owns a range of output tile columns and keeps the whole activation
// row resident in L1; the weights then stream past it column by column.
//
// The activation is **multicast**: one core reads it from DRAM and broadcasts it
// to every participating core's landing buffer. Read per core instead, it was
// 2.1 MB of the 6.9 MB the gate|up projection moves, and it is what capped the
// core count -- more cores meant more duplicate reads. `ttnn`'s own matmul does
// this (`mcast_in0=True`); a `generic_op` has to do it itself.
//
// The weight reads are **split with the writer kernel**. A core pulls about
// 15 GB/s over one NOC, and thirteen of them saturate at ~190 GB/s -- half the
// card. The writer is a second dataflow core with its own NOC and almost nothing
// to do (one tile a column), so it fetches the second half of every column into
// its own circular buffer. Same bytes, twice the ports. This reader takes
// k-tiles [0, H0).
//
// MODE 0 (the down projection, experts stacked on rows):
//     slot = kt / KT_E,  kt_e = kt % KT_E
//     w page = e*TPE + kt_e*NT_E + nt
// MODE 2 (gate|up, each expert's gate half ahead of every up half, as
//         `expert_gather.cpp` WIDE=2 lays it out):
//     c < K_SEL*HALF :  slot = c/HALF,        nt_e = c%HALF
//     otherwise      :  slot = (c-K_SEL*HALF)/HALF, nt_e = HALF + (c-...)%HALF
//     w page = e*TPE + kt*NT_E + nt_e
//
// Compile-time args:
//   0 KT        k-tiles in the reduction
//   1 KT_E      k-tiles in one expert (MODE 0 only; 0 otherwise)
//   2 NT_E      tiles across one expert's output
//   3 TPE       tiles in one expert
//   4 HALF      NT_E/2 (MODE 2 only)
//   5 K_SEL     selection width
//   6 MODE      0 rows, 2 gate|up split
//   7 NUM_EXPERTS
//   8 IDX16     1: uint16 straight out of ttnn.topk's tile face
//   9 A_PAGE    activation page bytes
//   10 W_PAGE    weight page bytes
//   11 H0        k-tiles this kernel fetches (the writer takes the rest)
//   12 N_DEST    cores in the multicast rectangle (the sender included)
//   13.. TensorAccessorArgs for a, w, idx
//
// Runtime args: 0 a_addr, 1 w_addr, 2 idx_addr, 3 col_lo, 4 col_hi,
//               5 is_sender, 6 mx0, 7 my0, 8 mx1, 9 my1, 10 sx, 11 sy

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t KT_E = get_compile_time_arg_val(1);
    constexpr uint32_t NT_E = get_compile_time_arg_val(2);
    constexpr uint32_t TPE = get_compile_time_arg_val(3);
    constexpr uint32_t HALF = get_compile_time_arg_val(4);
    constexpr uint32_t K_SEL = get_compile_time_arg_val(5);
    constexpr uint32_t MODE = get_compile_time_arg_val(6);
    constexpr uint32_t NUM_EXPERTS = get_compile_time_arg_val(7);
    constexpr uint32_t IDX16 = get_compile_time_arg_val(8);
    constexpr uint32_t A_PAGE = get_compile_time_arg_val(9);
    constexpr uint32_t W_PAGE = get_compile_time_arg_val(10);
    constexpr uint32_t H0 = get_compile_time_arg_val(11);
    constexpr uint32_t N_DEST = get_compile_time_arg_val(12);

    constexpr uint32_t cb_a = 0, cb_b = 1, cb_idx = 2;

    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t idx_addr = get_arg_val<uint32_t>(2);
    const uint32_t col_lo = get_arg_val<uint32_t>(3);
    const uint32_t col_hi = get_arg_val<uint32_t>(4);

    constexpr auto a_ta = TensorAccessorArgs<13>();
    const auto a_acc = TensorAccessor(a_ta, a_addr);
    constexpr auto w_ta = TensorAccessorArgs<a_ta.next_compile_time_args_offset()>();
    const auto w_acc = TensorAccessor(w_ta, w_addr);
    constexpr auto i_ta = TensorAccessorArgs<w_ta.next_compile_time_args_offset()>();
    const auto i_acc = TensorAccessor(i_ta, idx_addr);

    const uint32_t is_sender = get_arg_val<uint32_t>(5);
    const uint32_t mx0 = get_arg_val<uint32_t>(6);
    const uint32_t my0 = get_arg_val<uint32_t>(7);
    const uint32_t mx1 = get_arg_val<uint32_t>(8);
    const uint32_t my1 = get_arg_val<uint32_t>(9);
    const uint32_t sx = get_arg_val<uint32_t>(10);
    const uint32_t sy = get_arg_val<uint32_t>(11);

    // An idle core still takes part in the handshake -- the sender counts every
    // core in the rectangle, and one that returned early would hang it.

    // The selection, once. 64-byte aligned: a Blackhole DRAM transfer needs
    // (local & 63) == (noc & 63) and a CB base is only L1-aligned.
    const uint32_t idx_l1 = (get_write_ptr(cb_idx) + 63u) & ~63u;
    noc_async_read_page(0, i_acc, idx_l1);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* idx32 =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);
    volatile tt_l1_ptr uint16_t* idx16 =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(idx_l1);

    // The activation row, once for the whole rectangle, and resident: the
    // compute kernel indexes it by tile rather than popping it.
    cb_reserve_back(cb_a, KT);
    const uint32_t ab = get_write_ptr(cb_a);
    volatile tt_l1_ptr uint32_t* ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    volatile tt_l1_ptr uint32_t* valid =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    if (is_sender) {
        for (uint32_t kt = 0; kt < KT; ++kt) {
            noc_async_read_page(kt, a_acc, ab + kt * A_PAGE);
        }
        noc_async_read_barrier();
        noc_semaphore_wait(ready, N_DEST - 1);
        noc_semaphore_set(ready, 0);
        const uint64_t dst = get_noc_multicast_addr(mx0, my0, mx1, my1, ab);
        noc_async_write_multicast_loopback_src(ab, dst, KT * A_PAGE, N_DEST);
        noc_async_write_barrier();
        noc_semaphore_set(valid, 1);
        const uint64_t sdst =
            get_noc_multicast_addr(mx0, my0, mx1, my1, (uint32_t)get_semaphore(1));
        noc_semaphore_set_multicast_loopback_src(
            (uint32_t)get_semaphore(1), sdst, N_DEST);
    } else {
        noc_semaphore_set(valid, 0);
        noc_semaphore_inc(get_noc_addr(sx, sy, (uint32_t)get_semaphore(0)), 1);
        noc_semaphore_wait(valid, 1);
    }
    cb_push_back(cb_a, KT);

    if (col_lo >= col_hi) {
        return;                      // nothing else for an idle core to do
    }

    for (uint32_t c = col_lo; c < col_hi; ++c) {
        cb_reserve_back(cb_b, H0);
        const uint32_t bb = get_write_ptr(cb_b);

        uint32_t nt_e = 0, slot_c = 0;
        if constexpr (MODE == 2) {
            constexpr uint32_t SPLIT = K_SEL * HALF;
            if (c < SPLIT) {
                slot_c = c / HALF;
                nt_e = c - slot_c * HALF;
            } else {
                const uint32_t c2 = c - SPLIT;
                slot_c = c2 / HALF;
                nt_e = HALF + (c2 - slot_c * HALF);
            }
        } else {
            nt_e = c;
        }

        for (uint32_t kt = 0; kt < H0; ++kt) {
            uint32_t slot = slot_c;
            uint32_t kt_e = kt;
            if constexpr (MODE == 0) {
                slot = kt / KT_E;
                kt_e = kt - slot * KT_E;
            }
            const uint32_t raw = IDX16 ? (uint32_t)idx16[slot] : idx32[slot];
            // Clamped, not asserted: ASSERT expands to `while (1) {}` under the
            // watcher, and a hang costs a device reset. A clamp stays in bounds
            // and shows up host-side as a mismatch.
            const uint32_t e = raw < NUM_EXPERTS ? raw : 0;
            noc_async_read_page(e * TPE + kt_e * NT_E + nt_e, w_acc, bb + kt * W_PAGE);
        }
        noc_async_read_barrier();
        cb_push_back(cb_b, H0);
    }
}
