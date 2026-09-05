// delta_tail_compute.cpp -- rms-norm a head, scale it, and gate it.
//
//   gated[h, d] = out[h, d] * rsqrt(mean_d(out[h]^2) + eps) * w[d] * sigmoid(z[h, d])
//
// The reduction spans TPH tiles, so it accumulates elementwise first and reduces
// across the 32 columns once. `sfpu_reduce<REDUCE_ROW>` puts each row's total in
// that row's column 0, which is the operand shape `mul_tiles_bcast_cols` wants.
//
// `EPS_BITS`, not `EPS`: tt-metal's llk_math_common_api.h defines `EPS` as a
// macro and a constexpr of that name fails to compile against their header.
//
// Compile-time args: 0 TPH, 1 RECIP_D bits (1/hd), 2 EPS_BITS
// Runtime args: 0 head_lo, 1 head_hi

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
    constexpr uint32_t TPH = get_compile_time_arg_val(0);
    constexpr uint32_t RECIP_D = get_compile_time_arg_val(1);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(2);

    constexpr uint32_t cb_o = 0, cb_z = 1, cb_w = 2, cb_scale = 3, cb_out = 4,
                   cb_mid = 5;

    const uint32_t head_lo = get_arg_val<uint32_t>(0);
    const uint32_t head_hi = get_arg_val<uint32_t>(1);

    init_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(cb_o, cb_scale, cb_out);

    for (uint32_t h = head_lo; h < head_hi; ++h) {
        cb_wait_front(cb_o, TPH);
        cb_wait_front(cb_z, TPH);
        cb_wait_front(cb_w, TPH);

        init_sfpu(cb_o, cb_scale);
        tile_regs_acquire();
        for (uint32_t j = 0; j < TPH; ++j) {
            copy_tile(cb_o, j, 0);
            mul_binary_tile_init();
            if (j == 0) {
                mul_binary_tile(0, 0, 2);
            } else {
                mul_binary_tile(0, 0, 1);
                add_binary_tile_init();
                add_binary_tile(2, 1, 2);
            }
        }
        sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
        sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(2, 1, 1);
        binop_with_scalar_tile_init();
        mul_unary_tile(2, RECIP_D);          // sum -> mean
        add_unary_tile(2, EPS_BITS);
        rsqrt_tile_init();
        rsqrt_tile(2);
        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(cb_scale, 1);
        pack_tile(2, cb_scale);
        cb_push_back(cb_scale, 1);
        tile_regs_release();

        // Two windows a tile, not one. The broadcast multiply is an FPU op and
        // the rest is SFPU, and switching between them *inside* one
        // `tile_regs_acquire` gives a wrong answer rather than an error -- the
        // kernel measured 2.32x and 1.04 relative against float64 when the two
        // were mixed. Each window stays in one mode and hands over through a
        // circular buffer.
        cb_wait_front(cb_scale, 1);
        for (uint32_t j = 0; j < TPH; ++j) {
            init_sfpu(cb_o, cb_mid);
            tile_regs_acquire();
            copy_tile(cb_o, j, 0);
            copy_tile(cb_w, j, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);                        // * weight
            copy_tile(cb_z, j, 1);
            sigmoid_tile_init();
            sigmoid_tile(1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);                        // * sigmoid(gate)
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_mid, 1);
            pack_tile(0, cb_mid);
            cb_push_back(cb_mid, 1);
            tile_regs_release();

            cb_wait_front(cb_mid, 1);
            mul_bcast_cols_init_short(cb_mid, cb_scale);
            tile_regs_acquire();
            mul_tiles_bcast_cols(cb_mid, cb_scale, 0, 0, 0); // * the norm scale
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);
            tile_regs_release();
            cb_pop_front(cb_mid, 1);
        }
        cb_pop_front(cb_scale, 1);
        cb_pop_front(cb_o, TPH);
        cb_pop_front(cb_z, TPH);
        cb_pop_front(cb_w, TPH);
    }
}
