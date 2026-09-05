// outer_add_compute.cpp -- state = decayed + kt (x) delta, a tile at a time.
//
// The outer product wants a column broadcast (kt down the rows) and a row
// broadcast (delta across the columns), and the FPU does one at a time, so it
// takes two windows with a tile of ones to turn kt's single column into a full
// tile. The add is SFPU and gets its own window: mixing an FPU broadcast and the
// SFPU inside one `tile_regs_acquire` returns a wrong answer silently
// (invariant 87).
//
//   window 1   m1[i,j] = ones[i,j] * kt[i,0]        -> kt[i] everywhere
//   window 2   m2[i,j] = m1[i,j]  * delta[0,j]      -> kt[i] * delta[j]
//   window 3   out     = m2 + decayed
//
// The two broadcasts are different **types**, and `..._init_short` assumes the
// full init it is shortening was for the same one. Every working broadcast
// kernel in this project opens with one `init_bcast` and then uses that type
// only; this is the first to alternate, and it hung until each window got its
// own full `init_bcast`.
//
// STAGE bisects it: 1 stops after the column broadcast, 2 after the row
// broadcast, 3 is the whole thing. Only hang-or-not is meaningful below 3.
//
// Compile-time args: 0 STAGE
// Runtime args: 0 len

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"

void kernel_main() {
    constexpr uint32_t cb_d = 0, cb_k = 1, cb_v = 2, cb_1 = 3,
                       cb_m1 = 4, cb_m2 = 5, cb_out = 16;

    constexpr uint32_t STAGE = get_compile_time_arg_val(0);
    const uint32_t len = get_arg_val<uint32_t>(0);

    if (len == 0) {
        return;
    }
    cb_wait_front(cb_1, 1);
    // STAGE 4: the row broadcast **alone**, against the same ones tile the
    // column broadcast uses. Stage 1 (column alone) runs and stage 2 (both)
    // hangs; this says whether the row broadcast is unusable here or whether it
    // is having two broadcast windows in one kernel.
    if constexpr (STAGE == 4) {
        for (uint32_t j = 0; j < len; ++j) {
            cb_wait_front(cb_v, 1);
            init_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::ROW>(cb_1, cb_v, cb_out);
            tile_regs_acquire();
            mul_tiles_bcast<BroadcastType::ROW>(cb_1, cb_v, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);
            tile_regs_release();
            cb_pop_front(cb_v, 1);
            cb_wait_front(cb_k, 1);
            cb_pop_front(cb_k, 1);
            cb_wait_front(cb_d, 1);
            cb_pop_front(cb_d, 1);
        }
        return;
    }
    for (uint32_t j = 0; j < len; ++j) {
        cb_wait_front(cb_k, 1);
        init_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(cb_1, cb_k, cb_m1);
        tile_regs_acquire();
        mul_tiles_bcast<BroadcastType::COL>(cb_1, cb_k, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(STAGE == 1 ? cb_out : cb_m1, 1);
        pack_tile(0, STAGE == 1 ? cb_out : cb_m1);
        cb_push_back(STAGE == 1 ? cb_out : cb_m1, 1);
        tile_regs_release();
        cb_pop_front(cb_k, 1);
        if constexpr (STAGE == 1) {
            cb_wait_front(cb_v, 1);
            cb_pop_front(cb_v, 1);
            cb_wait_front(cb_d, 1);
            cb_pop_front(cb_d, 1);
            continue;
        }

        cb_wait_front(cb_m1, 1);
        cb_wait_front(cb_v, 1);
        init_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::ROW>(cb_m1, cb_v, cb_m2);
        tile_regs_acquire();
        mul_tiles_bcast<BroadcastType::ROW>(cb_m1, cb_v, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(STAGE == 2 ? cb_out : cb_m2, 1);
        pack_tile(0, STAGE == 2 ? cb_out : cb_m2);
        cb_push_back(STAGE == 2 ? cb_out : cb_m2, 1);
        tile_regs_release();
        cb_pop_front(cb_m1, 1);
        cb_pop_front(cb_v, 1);
        if constexpr (STAGE == 2) {
            cb_wait_front(cb_d, 1);
            cb_pop_front(cb_d, 1);
            continue;
        }

        cb_wait_front(cb_m2, 1);
        cb_wait_front(cb_d, 1);
        init_sfpu(cb_m2, cb_out);
        tile_regs_acquire();
        copy_tile(cb_m2, 0, 0);
        copy_tile(cb_d, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
        cb_pop_front(cb_m2, 1);
        cb_pop_front(cb_d, 1);
    }
}
