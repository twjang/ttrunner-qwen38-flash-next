// group_scale_compute.cpp -- x * rsqrt-scale * weight, two windows a tile.
//
// The scale arrives as one element -- (0, 0) of its tile -- and
// `mul_tiles_bcast_scalar` is the FPU op that reads exactly that. The weight
// multiply is SFPU. They must not share a `tile_regs_acquire`: mixing the two
// returns a wrong answer with nothing warning (invariant 87).
//
// Runtime args: 0 lo, 1 hi

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"

void kernel_main() {
    constexpr uint32_t cb_x = 0, cb_w = 1, cb_s = 2, cb_mid = 3, cb_out = 4;

    const uint32_t lo = get_arg_val<uint32_t>(0);
    const uint32_t hi = get_arg_val<uint32_t>(1);

    init_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::SCALAR>(cb_x, cb_s, cb_mid);

    for (uint32_t c = lo; c < hi; ++c) {
        cb_wait_front(cb_x, 1);
        cb_wait_front(cb_s, 1);
        mul_tiles_bcast_scalar_init_short(cb_x, cb_s);
        tile_regs_acquire();
        mul_tiles_bcast_scalar(cb_x, cb_s, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_mid, 1);
        pack_tile(0, cb_mid);
        cb_push_back(cb_mid, 1);
        tile_regs_release();
        cb_pop_front(cb_x, 1);
        cb_pop_front(cb_s, 1);

        cb_wait_front(cb_mid, 1);
        cb_wait_front(cb_w, 1);
        init_sfpu(cb_mid, cb_out);
        tile_regs_acquire();
        copy_tile(cb_mid, 0, 0);
        copy_tile(cb_w, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
        cb_pop_front(cb_mid, 1);
        cb_pop_front(cb_w, 1);
    }
}
