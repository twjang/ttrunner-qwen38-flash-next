// swiglu_compute.cpp -- silu(gate) * up, one tile at a time, in registers.
//
// The whole point of the fusion: `gate` and `up` arrive in circular buffers,
// the product leaves in one, and the intermediate never touches DRAM. The
// four-op ttnn chain it replaces round-trips the full E=128 tensor four times.
//
// `silu_tile` is a real SFPU op (compute_kernel_api.h), so this is two register
// operations and not a hand-rolled x*sigmoid(x).
//
// Runtime arg 0 is the tile count, not a compile-time arg: the work split is
// balanced but not equal, so cores differ by one and a compile-time count
// would be wrong for some of them.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_gate = 0;
    constexpr uint32_t cb_up = 1;
    constexpr uint32_t cb_out = 2;

    init_sfpu(cb_gate, cb_out);

    for (uint32_t i = 0; i < n_tiles; ++i) {
        cb_wait_front(cb_gate, 1);
        cb_wait_front(cb_up, 1);

        tile_regs_acquire();
        copy_tile(cb_gate, 0, 0);
        copy_tile(cb_up, 0, 1);

        silu_tile_init();
        silu_tile(0);                    // silu(gate)

        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);        // silu(gate) * up

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();

        cb_pop_front(cb_gate, 1);
        cb_pop_front(cb_up, 1);
    }
}
