// expert_gather_checksum.cpp -- stage-1 MoE sparse-read bandwidth probe.
//
// One data-movement kernel per core. It reads a selection list out of an
// io_tensor (NOT out of a runtime arg), turns each selected local expert id
// into a run of DRAM tile pages of the [1, E, K, N] expert weight tensor, reads
// only those pages, folds them into two uint32 accumulators, and writes one
// page of results.
//
// Why the indices come from a tensor: a captured trace freezes the dispatch
// commands and therefore the runtime args, but not tensor CONTENTS. Reading the
// selection from L1-landed tensor data is what makes one capture serve any
// routing. See dossier / HANDOFF "the trace objection does not apply".
//
// Address arithmetic (page == tile for an interleaved TILE_LAYOUT tensor):
//
//   tile_id(e, kt, nt) = (e * Kt + kt) * Nt + nt
//                      = e * TILES_PER_EXPERT + t,   t in [0, Kt*Nt)
//
//   work index w in [0, K_SEL * TILES_PER_EXPERT):
//     slot    = w / TILES_PER_EXPERT          (compile-time divisor)
//     t       = w - slot * TILES_PER_EXPERT
//     tile_id = idx[slot] * TILES_PER_EXPERT + t
//
//   TensorAccessor then does the interleave itself:
//     bank = tile_id % NUM_DRAM_BANKS, bank_page = tile_id / NUM_DRAM_BANKS,
//     addr = base + bank_page * aligned_page_size + bank_to_dram_offset[bank].
//
// Positional compile-time args, in the order the host appends them:
//   0: TILES_PER_EXPERT   (Kt*Nt, e.g. 3200 for [.,.,2560,1280])
//   1: TILE_BYTES         (accessor's ALIGNED page size: 576 bf4_b / 1088 bf8_b)
//   2: READ_BATCH         (tile reads issued between barriers)
//   3: IDX_PAGE_BYTES     (aligned page size of the index tensor)
//   4: OUT_PAGE_BYTES     (aligned page size of the output tensor)
//   5: NUM_EXPERTS         (experts_per_device, 128 -- bounds check only)
//   6..: TensorAccessorArgs for weights, then indices, then output
//
// Per-core runtime args:
//   0: weights.buffer_address()
//   1: indices.buffer_address()
//   2: out.buffer_address()
//   3: work_lo    (first work index this core owns)
//   4: work_hi    (one past the last)
//   5: out_page   (this core's output page id)

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"

void kernel_main() {
    constexpr uint32_t TILES_PER_EXPERT = get_compile_time_arg_val(0);
    constexpr uint32_t TILE_BYTES = get_compile_time_arg_val(1);
    constexpr uint32_t READ_BATCH = get_compile_time_arg_val(2);
    constexpr uint32_t IDX_PAGE_BYTES = get_compile_time_arg_val(3);
    constexpr uint32_t OUT_PAGE_BYTES = get_compile_time_arg_val(4);
    constexpr uint32_t NUM_EXPERTS = get_compile_time_arg_val(5);

    static_assert(TILES_PER_EXPERT > 0, "TILES_PER_EXPERT must be non-zero");
    static_assert(TILE_BYTES % 64 == 0, "tile pages must be a multiple of DRAM_ALIGNMENT(64)");
    static_assert(IDX_PAGE_BYTES % 64 == 0, "index page must be a multiple of DRAM_ALIGNMENT(64)");
    static_assert(OUT_PAGE_BYTES % 16 == 0, "output page must be a multiple of the DRAM write alignment");

    constexpr uint32_t cb_w = tt::CBIndex::c_0;    // weight landing scratch
    constexpr uint32_t cb_aux = tt::CBIndex::c_1;  // index page + output page

    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t idx_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    const uint32_t work_lo = get_arg_val<uint32_t>(3);
    const uint32_t work_hi = get_arg_val<uint32_t>(4);
    const uint32_t out_page = get_arg_val<uint32_t>(5);

    constexpr auto w_ta = TensorAccessorArgs<6>();
    const auto w_acc = TensorAccessor(w_ta, w_addr);
    constexpr auto i_ta = TensorAccessorArgs<w_ta.next_compile_time_args_offset()>();
    const auto i_acc = TensorAccessor(i_ta, idx_addr);
    constexpr auto o_ta = TensorAccessorArgs<i_ta.next_compile_time_args_offset()>();
    const auto o_acc = TensorAccessor(o_ta, out_addr);

    // This kernel is the sole owner of both CBs, so their write pointers are a
    // stable L1 scratch -- nothing is reserved, pushed or popped.
    //
    // Both landing addresses are forced to 64 B. A Blackhole DRAM read requires
    // (local_addr & 63) == (noc_addr & 63); DRAM buffer bases and both tile
    // sizes are 64-aligned, but a CB base is only guaranteed L1-aligned (16 B).
    // Each CB is declared one page larger than needed to pay for the bump.
    const uint32_t aux_base = (get_write_ptr(cb_aux) + 63u) & ~63u;
    const uint32_t idx_l1 = aux_base;
    const uint32_t out_l1 = aux_base + IDX_PAGE_BYTES;  // stays 64-aligned
    const uint32_t w_l1 = (get_write_ptr(cb_w) + 63u) & ~63u;

    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_l1);
    for (uint32_t i = 0; i < OUT_PAGE_BYTES / 4; ++i) {
        out[i] = 0;  // a core with no work still writes a defined page
    }

    // The selection list, straight out of the index io_tensor.
    noc_async_read_page(0, i_acc, idx_l1);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);

    uint32_t sum_data = 0;  // depends on the bytes that actually landed
    uint32_t sum_tid = 0;   // depends only on WHICH tiles were visited
    uint32_t n_tiles = 0;

    uint32_t w = work_lo;
    while (w < work_hi) {
        const uint32_t slot = w / TILES_PER_EXPERT;
        const uint32_t t = w - slot * TILES_PER_EXPERT;
        // The one data-dependent expression in the whole kernel.
        const uint32_t expert_id = idx[slot];
        // A garbage index would read another tensor's bytes (silent) or a wild
        // address (watcher hang). Cheap guard, watcher-visible, free in release.
        ASSERT(expert_id < NUM_EXPERTS);
        const uint32_t expert_first_tile = expert_id * TILES_PER_EXPERT;

        uint32_t run = TILES_PER_EXPERT - t;  // rest of this expert's slab
        if (run > work_hi - w) {
            run = work_hi - w;
        }

        uint32_t tile = expert_first_tile + t;
        const uint32_t tile_end = tile + run;
        while (tile < tile_end) {
            uint32_t batch = tile_end - tile;
            if (batch > READ_BATCH) {
                batch = READ_BATCH;
            }
            for (uint32_t i = 0; i < batch; ++i) {
                noc_async_read_page(tile + i, w_acc, w_l1 + i * TILE_BYTES);
            }
            noc_async_read_barrier();
            // One L1 load per tile. Folding all 144/272 words would cost more
            // RISC-V cycles than the read itself and would measure the wrong
            // thing; one word per tile is enough to make the data load real and
            // keep the accumulator dependent on it. Accuracy is irrelevant here.
            for (uint32_t i = 0; i < batch; ++i) {
                sum_data += reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w_l1 + i * TILE_BYTES)[0];
                sum_tid += tile + i;
            }
            tile += batch;
            n_tiles += batch;
        }
        w += run;
    }

    out[0] = sum_data;
    out[1] = sum_tid;
    out[2] = n_tiles;

    noc_async_write_page(out_page, o_acc, out_l1);
    noc_async_write_barrier();
}
