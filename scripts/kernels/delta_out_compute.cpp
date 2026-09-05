// delta_out_compute.cpp -- the delta rule's tail, per head.
//
//     delta = (v - predicted) * beta          (beta is one scalar a head)
//     qk    = sum_d q[d] * k[d]               (one scalar a head)
//     out   = q_decayed + qk * delta
//
// Written as `out = q_decayed + (qk*beta) * (v - predicted)`, so the difference
// is computed once and the two scalars are combined into one. Otherwise `delta`
// would have two consumers -- the writer and this kernel -- and a circular
// buffer has room for exactly one.
//
// Both scalars go through `mul_tiles_bcast_scalar`, which reads element (0, 0)
// of its second operand, and `sfpu_reduce<REDUCE_ROW>` leaves row 0's total
// exactly there -- so nothing is ever spread across a tile. The FPU broadcast
// and the SFPU never share a `tile_regs_acquire` (invariant 87).
//
// Compile-time args: 0 TPH
// Runtime args: 0 head_lo, 1 head_hi

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"

void kernel_main() {
    constexpr uint32_t TPH = get_compile_time_arg_val(0);
    constexpr uint32_t cb_v = 0, cb_p = 1, cb_b = 2, cb_q = 3, cb_k = 4, cb_d = 5;
    constexpr uint32_t cb_dm = 6, cb_qk = 7, cb_delta = 8, cb_out = 9, cb_s = 10,
                   cb_t = 11;

    const uint32_t head_lo = get_arg_val<uint32_t>(0);
    const uint32_t head_hi = get_arg_val<uint32_t>(1);

    init_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::SCALAR>(cb_dm, cb_b, cb_delta);

    for (uint32_t h = head_lo; h < head_hi; ++h) {
        cb_wait_front(cb_v, TPH);
        cb_wait_front(cb_p, TPH);
        cb_wait_front(cb_b, 1);
        cb_wait_front(cb_q, TPH);
        cb_wait_front(cb_k, TPH);
        cb_wait_front(cb_d, TPH);

        // qk, and then the single scalar qk*beta.
        init_sfpu(cb_q, cb_qk);
        tile_regs_acquire();
        for (uint32_t j = 0; j < TPH; ++j) {
            copy_tile(cb_q, j, 0);
            copy_tile(cb_k, j, 1);
            mul_binary_tile_init();
            if (j == 0) {
                mul_binary_tile(0, 1, 2);
            } else {
                mul_binary_tile(0, 1, 3);
                add_binary_tile_init();
                add_binary_tile(2, 3, 2);
            }
        }
        sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
        sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(2, 1, 1);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_qk, 1);
        pack_tile(2, cb_qk);
        cb_push_back(cb_qk, 1);
        tile_regs_release();

        cb_wait_front(cb_qk, 1);
        mul_tiles_bcast_scalar_init_short(cb_qk, cb_b);
        tile_regs_acquire();
        mul_tiles_bcast_scalar(cb_qk, cb_b, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_s, 1);
        pack_tile(0, cb_s);
        cb_push_back(cb_s, 1);
        tile_regs_release();
        cb_wait_front(cb_s, 1);

        for (uint32_t j = 0; j < TPH; ++j) {
            // d = v - predicted, once.
            init_sfpu(cb_v, cb_dm);
            tile_regs_acquire();
            copy_tile(cb_v, j, 0);
            copy_tile(cb_p, j, 1);
            sub_binary_tile_init();
            sub_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_dm, 1);
            pack_tile(0, cb_dm);
            cb_push_back(cb_dm, 1);
            tile_regs_release();
            cb_wait_front(cb_dm, 1);

            // delta = d * beta, for the state update downstream.
            mul_tiles_bcast_scalar_init_short(cb_dm, cb_b);
            tile_regs_acquire();
            mul_tiles_bcast_scalar(cb_dm, cb_b, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_delta, 1);
            pack_tile(0, cb_delta);
            cb_push_back(cb_delta, 1);
            tile_regs_release();

            // out = q_decayed + (qk*beta) * d
            mul_tiles_bcast_scalar_init_short(cb_dm, cb_s);
            tile_regs_acquire();
            mul_tiles_bcast_scalar(cb_dm, cb_s, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_t, 1);
            pack_tile(0, cb_t);
            cb_push_back(cb_t, 1);
            tile_regs_release();

            cb_wait_front(cb_t, 1);
            init_sfpu(cb_t, cb_out);
            tile_regs_acquire();
            copy_tile(cb_t, 0, 0);
            copy_tile(cb_d, j, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);
            tile_regs_release();
            cb_pop_front(cb_t, 1);
            cb_pop_front(cb_dm, 1);
        }

        cb_pop_front(cb_qk, 1);
        cb_pop_front(cb_s, 1);
        cb_pop_front(cb_v, TPH);
        cb_pop_front(cb_p, TPH);
        cb_pop_front(cb_b, 1);
        cb_pop_front(cb_q, TPH);
        cb_pop_front(cb_k, TPH);
        cb_pop_front(cb_d, TPH);
    }
}
