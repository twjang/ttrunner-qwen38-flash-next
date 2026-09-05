// small_ew_compute.cpp -- one elementwise op, chosen at compile time.
//
// The point of this kernel is not the arithmetic, it is the **core count**.
// ttnn's elementwise ops take the whole grid whatever the tensor -- a one-tile
// `ttnn.multiply` costs 5.78 us, which `dispatch_floor.py` shows is the price of
// a 110-core launch where one core is 2.06 -- and half of this model's 5764
// calls a step touch 64 tiles or fewer.
//
// OP: 0 multiply, 1 add, 2 subtract, 3 sigmoid, 4 silu, 5 copy,
//     6 multiply by scalar, 7 sigmoid then multiply by scalar
//
// Compile-time args: 0 OP, 1 SCALAR (float bits, for OP 6 and 7)
// Runtime args: 0 = tiles this core owns.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_binary_sfpu.h"

void kernel_main() {
    constexpr uint32_t OP = get_compile_time_arg_val(0);
    constexpr uint32_t SCALAR = get_compile_time_arg_val(1);
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_a = 0;
    constexpr uint32_t cb_b = 1;
    constexpr uint32_t cb_out = 2;
    constexpr bool BINARY = (OP <= 2);
    // Unary ops work in place at dst 0 and pack from there; only the binaries
    // need a third register, which avoids `copy_dest_values` and its header.
    constexpr uint32_t DST_OUT = BINARY ? 2 : 0;

    init_sfpu(cb_a, cb_out);

    for (uint32_t i = 0; i < n_tiles; ++i) {
        cb_wait_front(cb_a, 1);
        if (BINARY) {
            cb_wait_front(cb_b, 1);
        }
        tile_regs_acquire();
        copy_tile(cb_a, 0, 0);
        if (BINARY) {
            copy_tile(cb_b, 0, 1);
            if (OP == 0) {
                mul_binary_tile_init();
                mul_binary_tile(0, 1, 2);
            } else if (OP == 1) {
                add_binary_tile_init();
                add_binary_tile(0, 1, 2);
            } else {
                sub_binary_tile_init();
                sub_binary_tile(0, 1, 2);
            }
        } else if (OP == 3) {
            sigmoid_tile_init();
            sigmoid_tile(0);
        } else if (OP == 4) {
            silu_tile_init();
            silu_tile(0);
        } else if (OP == 6) {
            binop_with_scalar_tile_init();
            mul_unary_tile(0, SCALAR);
        } else if (OP == 7) {
            sigmoid_tile_init();
            sigmoid_tile(0);
            binop_with_scalar_tile_init();
            mul_unary_tile(0, SCALAR);
        }
        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(DST_OUT, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();

        cb_pop_front(cb_a, 1);
        if (BINARY) {
            cb_pop_front(cb_b, 1);
        }
    }
}
