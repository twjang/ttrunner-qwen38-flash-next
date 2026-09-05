// ksplit_sum_compute.cpp -- add G tiles into one.
//
// Two destination registers plus the accumulator, which fits the four that
// fp32_dest_acc_en leaves (invariant 81). The accumulation is fp32 across all G
// partials where `ttnn.sum` rounds to the tensor's dtype as it goes, so this is
// nearer the truth than the op it replaces, not merely as near.
//
// Compile-time args: 0 GROUPS
// Runtime args: 0 work_lo, 1 work_hi

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"

void kernel_main() {
    constexpr uint32_t GROUPS = get_compile_time_arg_val(0);
    const uint32_t work_lo = get_arg_val<uint32_t>(0);
    const uint32_t work_hi = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_in = 0, cb_out = 1;

    init_sfpu(cb_in, cb_out);

    for (uint32_t w = work_lo; w < work_hi; ++w) {
        cb_wait_front(cb_in, GROUPS);
        tile_regs_acquire();
        copy_tile(cb_in, 0, 2);                 // the accumulator
        add_binary_tile_init();
        for (uint32_t g = 1; g < GROUPS; ++g) {
            copy_tile(cb_in, g, 0);
            add_binary_tile(2, 0, 2);
        }
        tile_regs_commit();
        cb_pop_front(cb_in, GROUPS);

        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(2, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
    }
}
