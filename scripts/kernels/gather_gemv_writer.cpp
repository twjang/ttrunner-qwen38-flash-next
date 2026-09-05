// gather_gemv_writer.cpp -- the output tile, and the second half of every
// column's weights.
//
// A core pulls about 15 GB/s over one NOC. This kernel runs on the other
// dataflow core with its own NOC and, as a pure writer, almost nothing to do --
// one tile a column against the reader's KT. Giving it k-tiles [H0, KT) doubles
// the weight bandwidth a core can bring to bear for free, and the activation,
// which is the part that is duplicated across cores, stays with the reader.
//
// It runs one column behind on the output so its own reads stay ahead of the
// compute: fetch column c, then write column c-1.
//
// Compile-time args: 0 KT, 1 H0, 2 KT_E, 3 NT_E, 4 TPE, 5 HALF, 6 K_SEL,
//                    7 MODE, 8 NUM_EXPERTS, 9 IDX16, 10 W_PAGE,
//                    11.. TensorAccessorArgs for out, w, idx
// Runtime args: 0 out_addr, 1 w_addr, 2 idx_addr, 3 col_lo, 4 col_hi

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t H0 = get_compile_time_arg_val(1);
    constexpr uint32_t KT_E = get_compile_time_arg_val(2);
    constexpr uint32_t NT_E = get_compile_time_arg_val(3);
    constexpr uint32_t TPE = get_compile_time_arg_val(4);
    constexpr uint32_t HALF = get_compile_time_arg_val(5);
    constexpr uint32_t K_SEL = get_compile_time_arg_val(6);
    constexpr uint32_t MODE = get_compile_time_arg_val(7);
    constexpr uint32_t NUM_EXPERTS = get_compile_time_arg_val(8);
    constexpr uint32_t IDX16 = get_compile_time_arg_val(9);
    constexpr uint32_t W_PAGE = get_compile_time_arg_val(10);
    constexpr uint32_t H1 = KT - H0;

    constexpr uint32_t cb_b1 = 3, cb_idx2 = 4, cb_out = 16;

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t idx_addr = get_arg_val<uint32_t>(2);
    const uint32_t col_lo = get_arg_val<uint32_t>(3);
    const uint32_t col_hi = get_arg_val<uint32_t>(4);

    constexpr auto o_ta = TensorAccessorArgs<11>();
    const auto o_acc = TensorAccessor(o_ta, out_addr);
    constexpr auto w_ta = TensorAccessorArgs<o_ta.next_compile_time_args_offset()>();
    const auto w_acc = TensorAccessor(w_ta, w_addr);
    constexpr auto i_ta = TensorAccessorArgs<w_ta.next_compile_time_args_offset()>();
    const auto i_acc = TensorAccessor(i_ta, idx_addr);

    if (col_lo >= col_hi) {
        return;
    }

    const uint32_t idx_l1 = (get_write_ptr(cb_idx2) + 63u) & ~63u;
    noc_async_read_page(0, i_acc, idx_l1);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* idx32 =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);
    volatile tt_l1_ptr uint16_t* idx16 =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(idx_l1);

    for (uint32_t c = col_lo; c < col_hi; ++c) {
        if constexpr (H1 > 0) {
            cb_reserve_back(cb_b1, H1);
            const uint32_t bb = get_write_ptr(cb_b1);

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

            for (uint32_t j = 0; j < H1; ++j) {
                const uint32_t kt = H0 + j;
                uint32_t slot = slot_c;
                uint32_t kt_e = kt;
                if constexpr (MODE == 0) {
                    slot = kt / KT_E;
                    kt_e = kt - slot * KT_E;
                }
                const uint32_t raw = IDX16 ? (uint32_t)idx16[slot] : idx32[slot];
                const uint32_t e = raw < NUM_EXPERTS ? raw : 0;
                noc_async_read_page(e * TPE + kt_e * NT_E + nt_e, w_acc, bb + j * W_PAGE);
            }
            noc_async_read_barrier();
            cb_push_back(cb_b1, H1);
        }

        // One column behind, so this kernel's reads stay ahead of the compute.
        if (c > col_lo) {
            cb_wait_front(cb_out, 1);
            noc_async_write_page(c - 1, o_acc, get_read_ptr(cb_out));
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
    }
    cb_wait_front(cb_out, 1);
    noc_async_write_page(col_hi - 1, o_acc, get_read_ptr(cb_out));
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
}
