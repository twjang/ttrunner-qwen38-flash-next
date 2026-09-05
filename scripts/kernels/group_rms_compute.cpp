// group_rms_compute.cpp -- a partial sum of squares, or the scale it folds into.
//
// SQUARE=1: acc = sum_j x_j * x_j, elementwise across the run, packed as is.
// SQUARE=0: acc = sum_j p_j, then reduced across the 32 columns, scaled by 1/H,
//           offset by eps and rsqrt-ed. `sfpu_reduce<REDUCE_ROW>` leaves row 0's
//           total at element (0, 0), which is what `mul_tiles_bcast_scalar`
//           reads in the second kernel -- so no tile-wide spread is ever built.
//
// EPS_BITS, not EPS: tt-metal's llk headers define EPS as a macro.
//
// Compile-time args: 0 SQUARE, 1 RECIP_H bits, 2 EPS_BITS, 3 SPLIT_NUM, 4 SPLIT_DEN
// Runtime args: 0 work_lo, 1 work_hi, 2 run_len

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_binary_sfpu.h"

void kernel_main() {
    constexpr uint32_t SQUARE = get_compile_time_arg_val(0);
    constexpr uint32_t RECIP_H = get_compile_time_arg_val(1);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(2);
    constexpr uint32_t SPLIT_NUM = get_compile_time_arg_val(3);
    constexpr uint32_t SPLIT_DEN = get_compile_time_arg_val(4);

    constexpr uint32_t cb_x = 0, cb_x2 = 1, cb_out = 2;

    const uint32_t work_lo = get_arg_val<uint32_t>(0);
    const uint32_t work_hi = get_arg_val<uint32_t>(1);
    const uint32_t run_len = get_arg_val<uint32_t>(2);

    init_sfpu(cb_x, cb_out);
    if (work_lo >= work_hi) {
        return;
    }
    const uint32_t h0 = (run_len * SPLIT_NUM) / SPLIT_DEN;
    const uint32_t h1 = run_len - h0;

    cb_wait_front(cb_x, h0);
    if (h1 > 0) {
        cb_wait_front(cb_x2, h1);
    }
    tile_regs_acquire();
    bool first = true;
    for (uint32_t j = 0; j < h0; ++j) {
        copy_tile(cb_x, j, 0);
        if constexpr (SQUARE) {
            mul_binary_tile_init();
            if (first) {
                mul_binary_tile(0, 0, 2);
                first = false;
                continue;
            }
            mul_binary_tile(0, 0, 1);
        } else {
            if (first) {
                copy_tile(cb_x, j, 2);
                first = false;
                continue;
            }
            copy_tile(cb_x, j, 1);
        }
        add_binary_tile_init();
        add_binary_tile(2, 1, 2);
    }
    for (uint32_t j = 0; j < h1; ++j) {
        copy_tile(cb_x2, j, 0);
        if constexpr (SQUARE) {
            mul_binary_tile_init();
            if (first) {
                mul_binary_tile(0, 0, 2);
                first = false;
                continue;
            }
            mul_binary_tile(0, 0, 1);
        } else {
            if (first) {
                copy_tile(cb_x2, j, 2);
                first = false;
                continue;
            }
            copy_tile(cb_x2, j, 1);
        }
        add_binary_tile_init();
        add_binary_tile(2, 1, 2);
    }
    if constexpr (!SQUARE) {
        sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
        sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(2, 1, 1);
        binop_with_scalar_tile_init();
        mul_unary_tile(2, RECIP_H);
        add_unary_tile(2, EPS_BITS);
        rsqrt_tile_init();
        rsqrt_tile(2);
    }
    tile_regs_commit();

    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(2, cb_out);
    cb_push_back(cb_out, 1);
    tile_regs_release();

    cb_pop_front(cb_x, h0);
    if (h1 > 0) {
        cb_pop_front(cb_x2, h1);
    }
}
