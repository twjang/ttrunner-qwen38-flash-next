// gated_mean_compute.cpp -- sum_h (mix_h * normed_h) / hc, in registers.
//
// The nine ttnn ops this replaces each round-trip their operands through DRAM.
// Here the four products and the three adds live in the destination registers
// and only the final tile is packed out.
//
// Runtime args: 0 = output tiles this core owns.
// Compile-time args: 0 = HC, 1 = 1/hc as float bits (for mul_unary_tile).

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_binary_sfpu.h"

void kernel_main() {
    constexpr uint32_t HC = get_compile_time_arg_val(0);
    constexpr uint32_t INV_HC_BITS = get_compile_time_arg_val(1);
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_a = 0;
    constexpr uint32_t cb_b = 1;
    constexpr uint32_t cb_out = 2;

    init_sfpu(cb_a, cb_out);

    for (uint32_t i = 0; i < n_tiles; ++i) {
        cb_wait_front(cb_a, HC);
        cb_wait_front(cb_b, HC);

        tile_regs_acquire();
        for (uint32_t h = 0; h < HC; ++h) {
            copy_tile(cb_a, h, 0);
            copy_tile(cb_b, h, 1);
            mul_binary_tile_init();
            if (h == 0) {
                mul_binary_tile(0, 1, 4);            // accumulator starts here
            } else {
                mul_binary_tile(0, 1, 2);
                add_binary_tile_init();
                add_binary_tile(4, 2, 4);
            }
        }
        binop_with_scalar_tile_init();
        mul_unary_tile(4, INV_HC_BITS);              // the mean

        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(4, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();

        cb_pop_front(cb_a, HC);
        cb_pop_front(cb_b, HC);
    }
}
