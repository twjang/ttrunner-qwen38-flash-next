// expert_gather.cpp -- gather the selected experts' weight tiles into a compact
// tensor, driven by an index tensor rather than by runtime args.
//
// Stage 1 (expert_gather_checksum.cpp) proved the read side: only the selected
// experts' pages are fetched, at 246-389 GB/s, 1.87 ms for 48 layers against
// 30.8 for the whole of expert_ffn today. This is the same reader with the
// checksum replaced by a write, which is the part that makes it usable.
//
// What it buys: once the K_SEL selected experts sit side by side in one wide
// [K, K_SEL*N] tensor, the arithmetic is a single ordinary `ttnn.linear`. That
// removes four things at once -- `sparse_matmul`'s unconditional zero-fill of a
// [1, E, M, N] output (1.51 GB/token, handoff 5.2), the SwiGLU chain running
// over tensors that are 97 % zeros (handoff 5.4), the sparse op itself, and the
// separate combine -- without anyone having to write a matmul kernel.
//
// Why the indices come from a tensor: a captured trace freezes the dispatch
// commands and therefore the runtime args, but not tensor CONTENTS. Reading the
// selection from L1-landed tensor data is what lets one capture serve any
// routing (handoff 4d.1).
//
// Address arithmetic (page == tile for an interleaved TILE_LAYOUT tensor).
// Source tile, in the [1, E, K, N] weight tensor:
//
//   src(slot, t) = idx[slot] * TILES_PER_EXPERT + t
//
// Destination tile. The compact form would be [1, K_SEL, K, N] and `dst = w`,
// but that is not the layout worth having: a batched matmul over K_SEL experts
// at M=1 still cannot fill the core grid, and stage 2 measured it *slower* than
// the sparse matmul it replaces (0.382 ms against 0.321).
//
// Concatenating the experts on the **output** axis instead gives one wide
// [K, K_SEL*N] matmul -- the shape attn_qkv already runs at 128 GB/s -- and it
// measures 0.0479 ms against 0.321. So with `t = kt * NT + nt`:
//
//   dst(slot, t) = kt * (K_SEL * NT) + slot * NT + nt
//
// The down projection wants the other concatenation, on its *input* axis:
// [K_SEL*N, K], so that the matmul sums over the experts -- which is the
// combine, for free. Stacking [N, K] slabs on rows is exactly `dst = w`, the
// straightforward compact layout, so one flag covers both:
//
//   WIDE=1 (plain wide):  dst = kt * (K_SEL * NT) + slot * NT + nt
//   WIDE=0 (down):        dst = w
//   WIDE=2 (gate|up):     as 1, but with every expert's gate half moved ahead of
//                         every expert's up half, so the fused SwiGLU kernel --
//                         which splits its input down the middle -- works on the
//                         result unchanged. With HALF = NT/2:
//                           nt <  HALF: dst = kt*(K_SEL*NT) + slot*HALF + nt
//                           nt >= HALF: dst = kt*(K_SEL*NT) + K_SEL*HALF
//                                             + slot*HALF + (nt - HALF)
//
// Positional compile-time args, in the order the host appends them:
//   0: TILES_PER_EXPERT   (Kt*Nt)
//   1: TILE_BYTES         (accessor's ALIGNED page size)
//   2: READ_BATCH         (tiles in flight between barriers)
//   3: IDX_PAGE_BYTES     (aligned page size of the index tensor)
//   4: NUM_EXPERTS        (experts_per_device -- bounds clamp only)
//   5: K_SEL              (selection width; also separates program-cache entries,
//                          since generic_op hashes compile-time args by value but
//                          runtime args only by count -- handoff 4g)
//   6: NT                 (tiles across one expert's output, N/32)
//   7: WIDE               (1: concatenate on the output axis, for gate/up;
//                          0: stack on rows, for the down projection;
//                          2: as 1 with the gate|up halves split)
//   8: IDX16              (1: the selection is uint16 straight out of
//                          `ttnn.topk`, read from the first face of its
//                          tile -- which holds columns 0..15 contiguously,
//                          so k_sel <= 16 needs no typecast, no layout
//                          change and no pad. 0: a uint32 row-major page.)
//   9..: TensorAccessorArgs for weights, then indices, then output
//
// Per-core runtime args:
//   0: weights.buffer_address()
//   1: indices.buffer_address()
//   2: out.buffer_address()
//   3: work_lo    (first work index this core owns)
//   4: work_hi    (one past the last)

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t TILES_PER_EXPERT = get_compile_time_arg_val(0);
    constexpr uint32_t TILE_BYTES = get_compile_time_arg_val(1);
    constexpr uint32_t READ_BATCH = get_compile_time_arg_val(2);
    constexpr uint32_t IDX_PAGE_BYTES = get_compile_time_arg_val(3);
    constexpr uint32_t NUM_EXPERTS = get_compile_time_arg_val(4);
    constexpr uint32_t K_SEL = get_compile_time_arg_val(5);
    constexpr uint32_t NT = get_compile_time_arg_val(6);
    constexpr uint32_t WIDE = get_compile_time_arg_val(7);
    constexpr uint32_t IDX16 = get_compile_time_arg_val(8);
    constexpr uint32_t OUT_ROW_TILES = K_SEL * NT;

    static_assert(TILES_PER_EXPERT > 0, "TILES_PER_EXPERT must be non-zero");
    static_assert(TILE_BYTES % 64 == 0, "tile pages must be a multiple of DRAM_ALIGNMENT(64)");
    static_assert(IDX_PAGE_BYTES % 64 == 0, "index page must be a multiple of DRAM_ALIGNMENT(64)");

    constexpr uint32_t cb_w = 0;    // tile landing scratch
    constexpr uint32_t cb_aux = 1;  // the index page

    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t idx_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    const uint32_t work_lo = get_arg_val<uint32_t>(3);
    const uint32_t work_hi = get_arg_val<uint32_t>(4);

    constexpr auto w_ta = TensorAccessorArgs<9>();
    const auto w_acc = TensorAccessor(w_ta, w_addr);
    constexpr auto i_ta = TensorAccessorArgs<w_ta.next_compile_time_args_offset()>();
    const auto i_acc = TensorAccessor(i_ta, idx_addr);
    constexpr auto o_ta = TensorAccessorArgs<i_ta.next_compile_time_args_offset()>();
    const auto o_acc = TensorAccessor(o_ta, out_addr);

    // Sole owner of both CBs, so their write pointers are stable L1 scratch --
    // nothing is reserved, pushed or popped. Both landing addresses are forced
    // to 64 B: a Blackhole DRAM transfer requires (local & 63) == (noc & 63),
    // and a CB base is only guaranteed L1-aligned (16 B).
    const uint32_t idx_l1 = (get_write_ptr(cb_aux) + 63u) & ~63u;
    const uint32_t w_l1 = (get_write_ptr(cb_w) + 63u) & ~63u;

    noc_async_read_page(0, i_acc, idx_l1);
    noc_async_read_barrier();
    // Either a uint32 row-major page, or the raw first face of a uint16 tile
    // straight out of `ttnn.topk` -- which is what lets the host skip a
    // typecast, a layout change and a pad, three ops a layer.
    volatile tt_l1_ptr uint32_t* idx32 =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);
    volatile tt_l1_ptr uint16_t* idx16 =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(idx_l1);

    uint32_t w = work_lo;
    while (w < work_hi) {
        const uint32_t slot = w / TILES_PER_EXPERT;
        const uint32_t t = w - slot * TILES_PER_EXPERT;
        // The one data-dependent expression in the whole kernel.
        const uint32_t expert_id = IDX16 ? (uint32_t)idx16[slot] : idx32[slot];
        // Clamped rather than asserted: ASSERT expands to `while (1) { ; }`
        // under WATCHER_ENABLED, so a guard against a bad index would hang the
        // card, and a hang costs a device reset. A clamped read stays in bounds
        // and shows up host-side as a mismatch against the torch gather.
        const uint32_t safe_id = expert_id < NUM_EXPERTS ? expert_id : 0;

        uint32_t run = TILES_PER_EXPERT - t;          // rest of this expert's slab
        if (run > work_hi - w) {
            run = work_hi - w;
        }

        uint32_t src = safe_id * TILES_PER_EXPERT + t;
        const uint32_t end = src + run;
        uint32_t tt = t;                   // tile index within this expert
        while (src < end) {
            uint32_t batch = end - src;
            if (batch > READ_BATCH) {
                batch = READ_BATCH;
            }
            for (uint32_t i = 0; i < batch; ++i) {
                noc_async_read_page(src + i, w_acc, w_l1 + i * TILE_BYTES);
            }
            noc_async_read_barrier();
            for (uint32_t i = 0; i < batch; ++i) {
                // t = kt * NT + nt, and the wide output puts expert `slot`'s
                // columns at offset slot * NT within a row of OUT_ROW_TILES.
                const uint32_t ti = tt + i;
                uint32_t dst;
                if (WIDE == 2) {
                    constexpr uint32_t HALF = NT / 2;
                    const uint32_t kt = ti / NT;
                    const uint32_t nt = ti - kt * NT;
                    dst = kt * OUT_ROW_TILES + slot * HALF
                          + (nt < HALF ? nt : (K_SEL * HALF) + (nt - HALF));
                } else if (WIDE == 1) {
                    const uint32_t kt = ti / NT;
                    const uint32_t nt = ti - kt * NT;
                    dst = kt * OUT_ROW_TILES + slot * NT + nt;
                } else {
                    dst = slot * TILES_PER_EXPERT + ti;
                }
                noc_async_write_page(dst, o_acc, w_l1 + i * TILE_BYTES);
            }
            // The write must land before the next batch overwrites the same L1.
            noc_async_write_barrier();
            src += batch;
            tt += batch;
        }
        w += run;
    }
}
