// delta_scalars_compute.cpp -- exp(A * softplus(a + dt)) and sigmoid(b), one tile.
//
// Both halves live in the same tile: `a` in columns [0, H) and `b` in [H, 2H).
// The decay chain runs over the whole tile and is only *read* from the first
// half, so `dt` and `A` need nothing beyond their own columns -- whatever the
// pad holds lands in columns the writer never looks at.
//
// Four destination registers, not eight, because fp32_dest_acc_en halves the
// file (invariant 81): the value under construction sits at 2, its operands at
// 0 and 1.
//
// Compile-time args: 0 SOFTPLUS_BETA, 1 SOFTPLUS_BETA_RECIP, 2 SOFTPLUS_THRESHOLD
//                    (float bits; ttnn's defaults are 1.0, 1.0, 20.0)

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/eltwise_binary_sfpu.h"

void kernel_main() {
    constexpr uint32_t SP_BETA = get_compile_time_arg_val(0);
    constexpr uint32_t SP_RECIP = get_compile_time_arg_val(1);
    constexpr uint32_t SP_THRESH = get_compile_time_arg_val(2);

    constexpr uint32_t cb_ab = 0, cb_dt = 1, cb_ad = 2, cb_out = 3;

    init_sfpu(cb_ab, cb_out);

    cb_wait_front(cb_ab, 1);
    cb_wait_front(cb_dt, 1);
    cb_wait_front(cb_ad, 1);

    // g_exp = exp(A * softplus(a + dt))
    tile_regs_acquire();
    copy_tile(cb_ab, 0, 0);
    copy_tile(cb_dt, 0, 1);
    add_binary_tile_init();
    add_binary_tile(0, 1, 2);
    softplus_tile_init();
    softplus_tile(2, SP_BETA, SP_RECIP, SP_THRESH);
    copy_tile(cb_ad, 0, 1);
    mul_binary_tile_init();
    mul_binary_tile(2, 1, 2);
    exp_tile_init();
    exp_tile(2);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(2, cb_out);
    cb_push_back(cb_out, 1);
    tile_regs_release();

    // beta = sigmoid(b), from the untouched tile
    tile_regs_acquire();
    copy_tile(cb_ab, 0, 0);
    sigmoid_tile_init();
    sigmoid_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);
    tile_regs_release();

    cb_pop_front(cb_ab, 1);
    cb_pop_front(cb_dt, 1);
    cb_pop_front(cb_ad, 1);
}
