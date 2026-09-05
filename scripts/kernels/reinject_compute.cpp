// reinject_compute.cpp -- hyper + branch * inject, in registers.
//
// Runtime args: 0 = tiles this core owns.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_branch = 0;
    constexpr uint32_t cb_bcast = 1;
    constexpr uint32_t cb_hyper = 2;
    constexpr uint32_t cb_out = 4;

    init_sfpu(cb_branch, cb_out);

    for (uint32_t i = 0; i < n_tiles; ++i) {
        cb_wait_front(cb_branch, 1);
        cb_wait_front(cb_bcast, 1);
        cb_wait_front(cb_hyper, 1);

        tile_regs_acquire();
        copy_tile(cb_branch, 0, 0);
        copy_tile(cb_bcast, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 2);
        copy_tile(cb_hyper, 0, 3);
        add_binary_tile_init();
        add_binary_tile(2, 3, 4);
        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(4, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();

        cb_pop_front(cb_branch, 1);
        cb_pop_front(cb_bcast, 1);
        cb_pop_front(cb_hyper, 1);
    }
}
