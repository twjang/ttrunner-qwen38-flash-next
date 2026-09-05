// qkv_heads_compute.cpp -- two l2 norms a head, in one pass over the tiles.
//
//   l2norm(x, eps, scale) = scale * x * rsqrt(sum(x^2) + eps)
//
// which is what `_l2norm` computes: `rms_norm(x, eps/d)` is
// sqrt(d) * x * rsqrt(sum + eps), and the caller's `scale / sqrt(d)` cancels the
// sqrt(d). One reduction, one rsqrt, and a broadcast multiply -- no `rms_norm`
// program, and no round trip to DRAM between the two.
//
// The sum spans TPH tiles, so it is accumulated elementwise first (column c of
// the accumulator holds sum_j x_j[0, c]^2) and reduced across the 32 columns
// once. `sfpu_reduce<REDUCE_ROW>` leaves each row's total in that row's column
// 0, which is exactly the operand shape `mul_tiles_bcast_cols` wants: it scales
// row r by element (r, 0), so row 0 -- the only live row at M = 1 -- gets its
// own norm and the pad rows get theirs, harmlessly.
//
// Four destination registers, not eight (invariant 81).
//
// Compile-time args: 0 TPH, 1 EPS_BITS, 2 SCALE_Q bits, 3 SCALE_K bits
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

namespace {

constexpr uint32_t cb_q = 0, cb_k = 1, cb_scale = 2, cb_out = 3;

template <uint32_t TPH>
inline void norm_one(uint32_t cb_in, uint32_t eps, uint32_t scale) {
    cb_wait_front(cb_in, TPH);

    // sum of squares, elementwise across the head's tiles, then across columns
    init_sfpu(cb_in, cb_scale);
    tile_regs_acquire();
    for (uint32_t j = 0; j < TPH; ++j) {
        copy_tile(cb_in, j, 0);
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
    add_unary_tile(2, eps);
    rsqrt_tile_init();
    rsqrt_tile(2);
    binop_with_scalar_tile_init();
    mul_unary_tile(2, scale);
    tile_regs_commit();

    tile_regs_wait();
    cb_reserve_back(cb_scale, 1);
    pack_tile(2, cb_scale);
    cb_push_back(cb_scale, 1);
    tile_regs_release();

    // scale every tile of the head by that one column
    cb_wait_front(cb_scale, 1);
    mul_bcast_cols_init_short(cb_in, cb_scale);
    for (uint32_t j = 0; j < TPH; ++j) {
        tile_regs_acquire();
        mul_tiles_bcast_cols(cb_in, cb_scale, j, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
    }
    cb_pop_front(cb_scale, 1);
    cb_pop_front(cb_in, TPH);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t TPH = get_compile_time_arg_val(0);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(1);
    constexpr uint32_t SCALE_Q = get_compile_time_arg_val(2);
    constexpr uint32_t SCALE_K = get_compile_time_arg_val(3);

    const uint32_t head_lo = get_arg_val<uint32_t>(0);
    const uint32_t head_hi = get_arg_val<uint32_t>(1);

    init_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(cb_q, cb_scale, cb_out);

    for (uint32_t h = head_lo; h < head_hi; ++h) {
        norm_one<TPH>(cb_q, EPS_BITS, SCALE_Q);
        norm_one<TPH>(cb_k, EPS_BITS, SCALE_K);
    }
}
