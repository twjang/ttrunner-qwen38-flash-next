// ksplit_compute.cpp -- accumulate this core's slice of the reduction.
//
// One output tile per core, accumulated over [kt_lo, kt_hi). `matmul_tiles`
// accumulates into the destination register, so the whole slice folds into one
// tile before anything is packed.
//
// Runtime arg 0 is the tile count, not a compile-time arg: the K split is
// balanced but not equal, so cores differ by one.

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "hostdevcommon/kernel_structs.h"

void kernel_main() {
    const uint32_t n_k = get_arg_val<uint32_t>(0);
    constexpr tt::CBIndex cb_a = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_b = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_b, cb_out);
    matmul_init(cb_a, cb_b);

    tile_regs_acquire();
    for (uint32_t i = 0; i < n_k; ++i) {
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_b, 1);
        matmul_tiles(cb_a, cb_b, 0, 0, 0);
        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_b, 1);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);
    tile_regs_release();
}
