// gnorm1_compute.cpp -- the three phases, gated by the writer's circular buffers.
//
// Phase 1 is SFPU, phase 3 alternates the FPU broadcast and the SFPU, and they
// never share a `tile_regs_acquire` (invariant 87). The scale arrives as one
// element -- (0,0) of its tile -- which is where `sfpu_reduce<REDUCE_ROW>` leaves
// the total and what `mul_tiles_bcast_scalar` reads.
//
// The gatherer packs the finished scale TWICE: once into cb_scale, which its own
// phase 3 consumes, and once into cb_bcast, which the writer multicasts into
// everyone else's cb_scale. A circular buffer has one consumer; giving the
// writer its own buffer is what keeps the two from sharing a pop.
//
// Compile-time args: 0 RECIP_H bits, 1 EPS_BITS
// Runtime args: 0 tile_len, 1 is_gatherer, 2 parts

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_binary_sfpu.h"

void kernel_main() {
    constexpr uint32_t RECIP_H = get_compile_time_arg_val(0);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(1);
    constexpr uint32_t cb_x = 0, cb_w = 1, cb_part = 2, cb_fold = 3,
                       cb_scale = 4, cb_mid = 5, cb_out = 6, cb_bcast = 8;

    const uint32_t len = get_arg_val<uint32_t>(0);
    const uint32_t is_gatherer = get_arg_val<uint32_t>(1);
    const uint32_t parts = get_arg_val<uint32_t>(2);

    init_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::SCALAR>(cb_x, cb_scale, cb_mid);
    if (len == 0) {
        return;
    }

    // phase 1: this core's partial sum of squares, elementwise across its run
    cb_wait_front(cb_x, len);
    init_sfpu(cb_x, cb_part);
    tile_regs_acquire();
    for (uint32_t j = 0; j < len; ++j) {
        copy_tile(cb_x, j, 0);
        mul_binary_tile_init();
        if (j == 0) {
            mul_binary_tile(0, 0, 2);
        } else {
            mul_binary_tile(0, 0, 1);
            add_binary_tile_init();
            add_binary_tile(2, 1, 2);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_part, 1);
    pack_tile(2, cb_part);
    cb_push_back(cb_part, 1);
    tile_regs_release();

    // phase 2: the gatherer folds what the writer collected for it
    if (is_gatherer) {
        cb_wait_front(cb_fold, parts);
        init_sfpu(cb_fold, cb_scale);
        tile_regs_acquire();
        for (uint32_t j = 0; j < parts; ++j) {
            if (j == 0) {
                copy_tile(cb_fold, 0, 2);
                continue;
            }
            copy_tile(cb_fold, j, 1);
            add_binary_tile_init();
            add_binary_tile(2, 1, 2);
        }
        sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
        sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(2, 1, 1);
        binop_with_scalar_tile_init();
        mul_unary_tile(2, RECIP_H);
        add_unary_tile(2, EPS_BITS);
        rsqrt_tile_init();
        rsqrt_tile(2);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_scale, 1);
        pack_tile(2, cb_scale);
        cb_push_back(cb_scale, 1);
        cb_reserve_back(cb_bcast, 1);
        pack_tile(2, cb_bcast);
        cb_push_back(cb_bcast, 1);
        tile_regs_release();
        cb_pop_front(cb_fold, parts);
    }

    // phase 3: x * scale, then * weight -- the tiles are still in L1 from phase 1
    cb_wait_front(cb_scale, 1);
    cb_wait_front(cb_w, len);
    for (uint32_t j = 0; j < len; ++j) {
        mul_tiles_bcast_scalar_init_short(cb_x, cb_scale);
        tile_regs_acquire();
        mul_tiles_bcast_scalar(cb_x, cb_scale, j, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_mid, 1);
        pack_tile(0, cb_mid);
        cb_push_back(cb_mid, 1);
        tile_regs_release();

        cb_wait_front(cb_mid, 1);
        init_sfpu(cb_mid, cb_out);
        tile_regs_acquire();
        copy_tile(cb_mid, 0, 0);
        copy_tile(cb_w, j, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
        cb_pop_front(cb_mid, 1);
    }
    cb_pop_front(cb_x, len);
    cb_pop_front(cb_w, len);
    cb_pop_front(cb_scale, 1);
}
