// f32pack_compute.cpp -- one SFPU copy a tile, packed into the output CB.
// Runtime args: 0 len
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

void kernel_main() {
    constexpr uint32_t cb_in = 0, cb_out = 16;
    const uint32_t len = get_arg_val<uint32_t>(0);
    init_sfpu(cb_in, cb_out);
    for (uint32_t j = 0; j < len; ++j) {
        cb_wait_front(cb_in, 1);
        tile_regs_acquire();
        copy_tile(cb_in, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
        cb_pop_front(cb_in, 1);
    }
}
