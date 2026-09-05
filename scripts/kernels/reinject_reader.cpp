// reinject_reader.cpp -- feed the fused hyper-connection write-back.
//
// `reinject` is hyper + (branch outer-product inject), flattened back to
// hc*hidden, and it runs **96 times a token**. As four ttnn ops -- a broadcast
// multiply, a permute, a reshape and an add -- it materialises a
// [.., hc, M, hidden] intermediate twice: at M=1 that tile stack is padded from
// one row to thirty-two, so each of those ops moves ~655 KB to carry 10 KB.
//
// One pass reads hyper and branch once and writes the result once. The permute
// disappears entirely, because a writer chooses its destination tile: there is
// no such thing as a layout change when you are already choosing where to put
// each tile.
//
//   out[m, h*HIDDEN + d] = hyper[m, h*HIDDEN + d] + branch[m, d] * inject[h, m]
//
// Tile addressing. The output and `hyper` share a layout of NT = HC*NT_H tiles
// a row-tile, so the output tile index *is* the work index:
//
//   w  = mt * NT + c,  h = c / NT_H,  dt = c % NT_H
//   hyper  tile = w
//   branch tile = mt * NT_H + dt          ([1, 1, M, HIDDEN])
//   inject page = h * MT + mt             ([1, HC, M, 1], one tile per (h, mt))
//
// `inject` carries one scalar per (h, m) and the SFPU multiplies whole tiles, so
// the reader synthesises a tile whose row m is filled with inject[h, m]. It is
// rebuilt only when (h, mt) changes, which is once every NT_H work items.
//
// Compile-time args:
//   0: NT_H            (HIDDEN / 32)
//   1: HC              (hyper-connection count)
//   2: MT              (row-tiles: ceil(M / 32))
//   3: TILE_BYTES      (aligned page size of hyper/branch/out -- one dtype)
//   4: INJ_TILE_BYTES  (aligned page size of the inject tensor)
//   5: BF16            (1: operands are bfloat16, 0: float32)
//   6: INJ_RAW         (1: `inject` is the *un-sliced* gate stream and column
//                        INJ_BASE + h of it is the scalar, which the compute
//                        kernel then puts through 2*sigmoid. 0: it is the
//                        [1, HC, M, 1] tensor the ttnn ops used to build, one
//                        page per (h, mt), value in column 0.)
//   7: INJ_BASE        (first gate column, when INJ_RAW)
//   8: INJ_NT          (tiles across a row of the gate stream, when INJ_RAW)
//   9..: TensorAccessorArgs for hyper, branch, inject
//
// INJ_RAW exists because building that [1, HC, M, 1] tensor cost a slice, a
// sigmoid, a multiply and a permute -- four ttnn ops, 96 times a token, on a
// stream four numbers wide -- and this kernel was already reading the tile they
// were built from.
//
// Runtime args: 0 hyper_addr, 1 branch_addr, 2 inject_addr, 3 work_lo, 4 work_hi

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t NT_H = get_compile_time_arg_val(0);
constexpr uint32_t HC = get_compile_time_arg_val(1);
constexpr uint32_t MT = get_compile_time_arg_val(2);
constexpr uint32_t TILE_BYTES = get_compile_time_arg_val(3);
constexpr uint32_t INJ_TILE_BYTES = get_compile_time_arg_val(4);
constexpr uint32_t BF16 = get_compile_time_arg_val(5);
constexpr uint32_t INJ_RAW = get_compile_time_arg_val(6);
constexpr uint32_t INJ_BASE = get_compile_time_arg_val(7);
constexpr uint32_t INJ_NT = get_compile_time_arg_val(8);
constexpr uint32_t NT = HC * NT_H;

// Element offset of (row, col) in a 32x32 tile laid out as four 16x16 faces.
inline uint32_t tile_off(uint32_t row, uint32_t col) {
    const uint32_t face = (row < 16 ? 0u : 2u) + (col < 16 ? 0u : 1u);
    return face * 256u + (row & 15u) * 16u + (col & 15u);
}

void kernel_main() {
    constexpr uint32_t cb_branch = 0;
    constexpr uint32_t cb_bcast = 1;
    constexpr uint32_t cb_hyper = 2;
    constexpr uint32_t cb_scratch = 3;      // the raw inject page

    const uint32_t hyper_addr = get_arg_val<uint32_t>(0);
    const uint32_t branch_addr = get_arg_val<uint32_t>(1);
    const uint32_t inject_addr = get_arg_val<uint32_t>(2);
    const uint32_t work_lo = get_arg_val<uint32_t>(3);
    const uint32_t work_hi = get_arg_val<uint32_t>(4);

    constexpr auto h_ta = TensorAccessorArgs<9>();
    const auto h_acc = TensorAccessor(h_ta, hyper_addr);
    constexpr auto b_ta = TensorAccessorArgs<h_ta.next_compile_time_args_offset()>();
    const auto b_acc = TensorAccessor(b_ta, branch_addr);
    constexpr auto i_ta = TensorAccessorArgs<b_ta.next_compile_time_args_offset()>();
    const auto i_acc = TensorAccessor(i_ta, inject_addr);

    // A Blackhole DRAM transfer needs (local & 63) == (noc & 63), and a CB base
    // is only L1-aligned, so the landing address is forced to 64 B.
    const uint32_t scratch = (get_write_ptr(cb_scratch) + 63u) & ~63u;

    // Which (page, column) the *broadcast* CB slot already holds. Work items run
    // consecutively in c, so h -- and with it the column -- changes once every
    // NT_H of them: rebuilding the spread on every item was 1024 scalar writes a
    // tile and cost more than the ops this kernel replaces.
    uint32_t have = 0xffffffffu;
    uint32_t have_col = 0xffffffffu;
    uint32_t bcast_l1 = 0;
    for (uint32_t w = work_lo; w < work_hi; ++w) {
        const uint32_t mt = w / NT;
        const uint32_t c = w - mt * NT;
        const uint32_t h = c / NT_H;
        const uint32_t dt = c - h * NT_H;

        // Which page holds the scalar, and which column of it.
        const uint32_t src_col = INJ_RAW ? ((INJ_BASE + h) & 31u) : 0u;
        const uint32_t which = INJ_RAW ? (mt * INJ_NT + ((INJ_BASE + h) >> 5))
                                       : (h * MT + mt);
        cb_reserve_back(cb_bcast, 1);
        const uint32_t bc = get_write_ptr(cb_bcast);
        const bool fresh = (which != have) || (src_col != have_col) || (bc != bcast_l1);
        if (which != have) {
            noc_async_read_page(which, i_acc, scratch);
            noc_async_read_barrier();
            have = which;
        }
        if (fresh) {
            have_col = src_col;
            bcast_l1 = bc;
        }
        // Spread that column across all 32, so the elementwise multiply
        // downstream sees a per-row scalar. Rebuilt every work item rather than
        // cached, because under INJ_RAW four consecutive h values share one
        // page and differ only in the column.
        if (!fresh) {
            // the CB slot already holds this scalar spread
        } else if (BF16) {
            volatile tt_l1_ptr uint16_t* src =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch);
            volatile tt_l1_ptr uint16_t* dst =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(bc);
            for (uint32_t r = 0; r < 32; ++r) {
                const uint16_t v = src[tile_off(r, src_col)];
                for (uint32_t col = 0; col < 32; ++col) {
                    dst[tile_off(r, col)] = v;
                }
            }
        } else {
            volatile tt_l1_ptr uint32_t* src =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
            volatile tt_l1_ptr uint32_t* dst =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bc);
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t v = src[tile_off(r, src_col)];
                for (uint32_t col = 0; col < 32; ++col) {
                    dst[tile_off(r, col)] = v;
                }
            }
        }

        cb_reserve_back(cb_branch, 1);
        noc_async_read_page(mt * NT_H + dt, b_acc, get_write_ptr(cb_branch));
        cb_reserve_back(cb_hyper, 1);
        noc_async_read_page(w, h_acc, get_write_ptr(cb_hyper));
        noc_async_read_barrier();

        cb_push_back(cb_branch, 1);
        cb_push_back(cb_bcast, 1);
        cb_push_back(cb_hyper, 1);
    }
}
