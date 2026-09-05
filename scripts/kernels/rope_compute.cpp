// rope_compute.cpp -- the two fused tiles, and a copy for the rest.
//
// Compile-time args: 0 NT, 1 NT_ROPE
// Runtime args: 0 work_lo, 1 work_hi

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"

void kernel_main() {
    constexpr uint32_t NT = get_compile_time_arg_val(0);
    constexpr uint32_t NT_ROPE = get_compile_time_arg_val(1);
    const uint32_t work_lo = get_arg_val<uint32_t>(0);
    const uint32_t work_hi = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_a = 0;
    constexpr uint32_t cb_b = 1;
    constexpr uint32_t cb_cos = 2;
    constexpr uint32_t cb_sin = 3;
    constexpr uint32_t cb_out = 4;

    init_sfpu(cb_a, cb_out);

    for (uint32_t w = work_lo; w < work_hi; ++w) {
        const uint32_t r = w / NT;
        const uint32_t c = w - r * NT;

        cb_wait_front(cb_a, 1);
        tile_regs_acquire();
        if (c < NT_ROPE) {
            cb_wait_front(cb_b, 1);
            cb_wait_front(cb_cos, 1);
            cb_wait_front(cb_sin, 1);
            // c == 0:  x0 * cos - x1 * sin
            // c == 1:  x1 * cos + x0 * sin
            //
            // Three destination registers, not seven. With fp32_dest_acc_en the
            // register file holds **four** tiles, not eight, and writing past
            // that is silent: it worked at one and eight sequences and gave
            // 1.08 relative error at thirty-two, where more tiles are in flight
            // (`rope_kernel_check.py`, and `batch_equivalence_check.py` went
            // 32/32 -> 16/32).
            copy_tile(cb_cos, 0, 2);
            copy_tile(cb_a, 0, c == 0 ? 0 : 1);
            copy_tile(cb_b, 0, c == 0 ? 1 : 0);
            mul_binary_tile_init();
            mul_binary_tile(0, 2, 0);          // (c==0 ? x0 : x1) * cos
            copy_tile(cb_sin, 0, 2);
            mul_binary_tile(1, 2, 1);          // (c==0 ? x1 : x0) * sin
            if (c == 0) {
                sub_binary_tile_init();
                sub_binary_tile(0, 1, 0);
            } else {
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
            }
        } else {
            copy_tile(cb_a, 0, 0);
        }
        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();

        cb_pop_front(cb_a, 1);
        if (c < NT_ROPE) {
            cb_pop_front(cb_b, 1);
            cb_pop_front(cb_cos, 1);
            cb_pop_front(cb_sin, 1);
        }
    }
}
