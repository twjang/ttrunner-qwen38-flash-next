// expert_gather.cpp -- gather the selected experts' weight tiles into a compact
// tensor, driven by an index tensor rather than by runtime args.
//
// Stage 1 (expert_gather_checksum.cpp) proved the read side: only the selected
// experts' pages are fetched, at 246-389 GB/s, 1.87 ms for 48 layers against
// 30.8 for the whole of expert_ffn today. This is the same reader with the
// checksum replaced by a write, which is the part that makes it usable.
//
// What it buys, and why it beats writing a full fused MoE kernel: once the
// weights of the K_SEL selected experts sit in a compact [1, K_SEL, K, N]
// tensor, the arithmetic is an ordinary dense batched `ttnn.matmul` over an
// expert axis of K_SEL instead of E. That removes three things at once --
// `sparse_matmul`'s unconditional zero-fill of a [1, E, M, N] output
// (1.51 GB/token, handoff 5.2), the SwiGLU chain running over tensors that are
// 97 % zeros (6.83 ms, handoff 5.4), and the sparse op itself -- without anyone
// having to write a matmul kernel.
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
// Destination tile, in the compact [1, K_SEL, K, N] output: the work index
// itself, because the output's expert axis is the slot axis --
//
//   dst(slot, t) = slot * TILES_PER_EXPERT + t = w
//
// so the gather is a permutation of whole expert slabs and needs no shuffling
// within one.
//
// Positional compile-time args, in the order the host appends them:
//   0: TILES_PER_EXPERT   (Kt*Nt)
//   1: TILE_BYTES         (accessor's ALIGNED page size)
//   2: READ_BATCH         (tiles in flight between barriers)
//   3: IDX_PAGE_BYTES     (aligned page size of the index tensor)
//   4: NUM_EXPERTS        (experts_per_device -- bounds clamp only)
//   5: K_SEL              (selection width; here only to separate program-cache
//                          entries, since generic_op hashes compile-time args by
//                          value but runtime args only by count -- handoff 4g)
//   6..: TensorAccessorArgs for weights, then indices, then output
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

    constexpr auto w_ta = TensorAccessorArgs<6>();
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
    volatile tt_l1_ptr uint32_t* idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);

    uint32_t w = work_lo;
    while (w < work_hi) {
        const uint32_t slot = w / TILES_PER_EXPERT;
        const uint32_t t = w - slot * TILES_PER_EXPERT;
        // The one data-dependent expression in the whole kernel.
        const uint32_t expert_id = idx[slot];
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
        uint32_t dst = w;
        const uint32_t end = src + run;
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
                noc_async_write_page(dst + i, o_acc, w_l1 + i * TILE_BYTES);
            }
            // The write must land before the next batch overwrites the same L1.
            noc_async_write_barrier();
            src += batch;
            dst += batch;
        }
        w += run;
    }
}
