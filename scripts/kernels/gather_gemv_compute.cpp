// gather_gemv_compute.cpp -- accumulate KT tile products into one output tile.
//
// The activation stays in its circular buffer for the whole core: it is indexed
// by tile, never popped, because every output column this core owns multiplies
// against the same row.
//
// Compile-time args: 0 KT
// Runtime args: 0 col_lo, 1 col_hi

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "hostdevcommon/kernel_structs.h"

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    const uint32_t col_lo = get_arg_val<uint32_t>(0);
    const uint32_t col_hi = get_arg_val<uint32_t>(1);

    constexpr tt::CBIndex cb_a = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_b = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_b, cb_out);
    if (col_lo >= col_hi) {
        return;
    }
    matmul_init(cb_a, cb_b);

    cb_wait_front(cb_a, KT);
    for (uint32_t c = col_lo; c < col_hi; ++c) {
        cb_wait_front(cb_b, KT);
        tile_regs_acquire();
        for (uint32_t i = 0; i < KT; ++i) {
            matmul_tiles(cb_a, cb_b, i, i, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
        cb_pop_front(cb_b, KT);
    }
}
