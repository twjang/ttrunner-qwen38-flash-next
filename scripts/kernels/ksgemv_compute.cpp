// ksgemv_compute.cpp -- this core's slice of the reduction, and the fold.
//
// Phase 1: accumulate [kt_lo, kt_lo+klen) into one destination register and pack
// it as a **float32** partial. The split changes the summation order, so the
// partials are the one place the precision can be given back for nothing.
//
// Phase 2, on the core that owns output tile n in group 0: add the G partials
// the writer collected and pack the output tile. FPU matmul and SFPU add never
// share a `tile_regs_acquire` (invariant 87).
//
// klen is a runtime arg because the split is balanced, not equal, and it is zero
// for a core no group covers -- such a core must pack nothing, or it hands the
// writer whatever its destination register happened to hold.
//
// Compile-time args: 0 FOLD (0 bisects the cross-core handshake out),
//                    1 RESET (redo the hardware startup before the SFPU window)
// Runtime args: 0 klen, 1 active, 2 is_gatherer, 3 G

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "hostdevcommon/kernel_structs.h"

void kernel_main() {
    constexpr uint32_t FOLD = get_compile_time_arg_val(0);
    constexpr uint32_t RESET = get_compile_time_arg_val(1);
    constexpr uint32_t cb_a = 0, cb_w = 1, cb_part = 2, cb_fold = 3, cb_out = 16;
    constexpr uint32_t cb_acc = FOLD ? cb_part : cb_out;

    const uint32_t klen = get_arg_val<uint32_t>(0);
    const uint32_t active = get_arg_val<uint32_t>(1);
    const uint32_t is_gatherer = get_arg_val<uint32_t>(2);
    const uint32_t g_count = get_arg_val<uint32_t>(3);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_w, cb_acc);
    if (klen == 0 || active == 0) {
        return;
    }
    matmul_init(cb_a, cb_w);
    cb_wait_front(cb_a, klen);
    cb_wait_front(cb_w, klen);
    tile_regs_acquire();
    for (uint32_t i = 0; i < klen; ++i) {
        matmul_tiles(cb_a, cb_w, i, i, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_acc, 1);
    pack_tile(0, cb_acc);
    cb_push_back(cb_acc, 1);
    tile_regs_release();
    cb_pop_front(cb_a, klen);
    cb_pop_front(cb_w, klen);

    if (!is_gatherer || !FOLD) {
        return;
    }
    cb_wait_front(cb_fold, g_count);
    // The matmul left the unpacker in SrcOrder::Reverse; nothing else in this
    // project follows a matmul window with an SFPU one, so the startup is redone
    // in the default order before the fold.
    if constexpr (RESET) {
        compute_kernel_hw_startup(cb_fold, cb_out);
    }
    init_sfpu(cb_fold, cb_out);
    tile_regs_acquire();
    for (uint32_t j = 0; j < g_count; ++j) {
        if (j == 0) {
            copy_tile(cb_fold, 0, 2);
            continue;
        }
        copy_tile(cb_fold, j, 1);
        add_binary_tile_init();
        add_binary_tile(2, 1, 2);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(2, cb_out);
    cb_push_back(cb_out, 1);
    tile_regs_release();
    cb_pop_front(cb_fold, g_count);
}
