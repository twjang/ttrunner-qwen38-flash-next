// conv_step_compute.cpp -- sum_j w[j] * hist[j], then silu, in one pass.
//
// tap 0 is the oldest column and tap 3 the newest, matching `_causal_conv_step`:
// age = (k-1) - tap, so tap 0 pairs with state[2] and tap 3 with the new column.
//
// Four destination registers, not eight: `fp32_dest_acc_en` halves the register
// file, and writing past index 3 is silent until the batch is wide enough to
// keep several tiles in flight (handoff 21.1). The accumulator lives at 2 and
// the two operands at 0 and 1, with 3 as the term being folded in.
//
// The accumulation is fp32 across all four taps, where the ops path rounds to
// bfloat16 between every multiply and add -- so this is nearer the truth, not
// merely equal to what it replaces.
//
// Runtime args: 0 work_lo, 1 work_hi

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"

void kernel_main() {
    const uint32_t work_lo = get_arg_val<uint32_t>(0);
    const uint32_t work_hi = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_x = 0, cb_s0 = 1, cb_s1 = 2, cb_s2 = 3;
    constexpr uint32_t cb_w0 = 4, cb_w1 = 5, cb_w2 = 6, cb_w3 = 7;
    constexpr uint32_t cb_out = 8;

    init_sfpu(cb_x, cb_out);

    for (uint32_t w = work_lo; w < work_hi; ++w) {
        cb_wait_front(cb_x, 1);
        cb_wait_front(cb_s0, 1);
        cb_wait_front(cb_s1, 1);
        cb_wait_front(cb_s2, 1);
        cb_wait_front(cb_w0, 1);
        cb_wait_front(cb_w1, 1);
        cb_wait_front(cb_w2, 1);
        cb_wait_front(cb_w3, 1);

        tile_regs_acquire();
        mul_binary_tile_init();
        copy_tile(cb_w0, 0, 0);
        copy_tile(cb_s2, 0, 1);
        mul_binary_tile(0, 1, 2);

        copy_tile(cb_w1, 0, 0);
        copy_tile(cb_s1, 0, 1);
        mul_binary_tile(0, 1, 3);
        add_binary_tile_init();
        add_binary_tile(2, 3, 2);

        mul_binary_tile_init();
        copy_tile(cb_w2, 0, 0);
        copy_tile(cb_s0, 0, 1);
        mul_binary_tile(0, 1, 3);
        add_binary_tile_init();
        add_binary_tile(2, 3, 2);

        mul_binary_tile_init();
        copy_tile(cb_w3, 0, 0);
        copy_tile(cb_x, 0, 1);
        mul_binary_tile(0, 1, 3);
        add_binary_tile_init();
        add_binary_tile(2, 3, 2);

        silu_tile_init();
        silu_tile(2);
        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(2, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();

        cb_pop_front(cb_x, 1);
        cb_pop_front(cb_s0, 1);
        cb_pop_front(cb_s1, 1);
        cb_pop_front(cb_s2, 1);
        cb_pop_front(cb_w0, 1);
        cb_pop_front(cb_w1, 1);
        cb_pop_front(cb_w2, 1);
        cb_pop_front(cb_w3, 1);
    }
}
