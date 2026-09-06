// fabreduce_compute.cpp -- `out = mine + theirs`, one tile at a time.
//
// SFPU, not a matmul: there is no contraction here, so none of 45.32's padding
// hazard applies -- an elementwise add reads only the elements it is given.
//
// Includes and entry point copied from `recur_compute.cpp`, which is the form
// that builds in this tree (`api/compute/...`, `void kernel_main()`); the
// upstream `compute_kernel_api/...` paths and `namespace NAMESPACE { void MAIN }`
// do not exist here.
//
// Compile-time args: 0 ROLE, 1 NTILES

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"   // init_sfpu
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_hw_startup.h"

void kernel_main() {
    constexpr uint32_t ROLE = get_compile_time_arg_val(0);
    constexpr uint32_t NT = get_compile_time_arg_val(1);
    constexpr uint32_t cb_mine = 0, cb_theirs = 1, cb_out = 16;

    compute_kernel_hw_startup(cb_mine, cb_theirs, cb_out);
    if constexpr (ROLE != 2 && ROLE != 3) {
        return;   // idle and the pure sender have nothing to add
    }
    init_sfpu(cb_mine, cb_out);
    for (uint32_t i = 0; i < NT; ++i) {
        cb_wait_front(cb_mine, 1);
        cb_wait_front(cb_theirs, 1);
        cb_reserve_back(cb_out, 1);
        tile_regs_acquire();
        copy_tile(cb_mine, 0, 0);
        copy_tile(cb_theirs, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 2);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(2, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_mine, 1);
        cb_pop_front(cb_theirs, 1);
    }
}
