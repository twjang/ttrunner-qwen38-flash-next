// group_rms_reader.cpp -- a run of tiles, for a partial sum of squares.
//
// `grouped_rms_norm` is six launches -- a reshape to put the groups on rows, the
// sharded norm's three, a reshape back and the weight multiply -- and
// `mesh_partition` behind it is a seventh: **26.72 us** together, 97 times a
// token. The reshapes exist only because `ttnn.rms_norm` reduces over the last
// axis. A kernel that does its own reduction leaves the groups where they are.
//
// The reduction runs in two passes because it is SFPU-bound, not
// bandwidth-bound: one core squaring and accumulating a whole 80-tile group is
// 240 tile operations and measured **31 us** on four cores. Spread over ten
// cores a group it is 24 each, and a second launch folds the partials.
//
// Pass 1 (SQUARE=1): core c owns tiles [lo, hi) of one group, writes one partial.
// Pass 2 (SQUARE=0): core g owns the partials of group g, writes one scale.
//
// The fetch is split with the writer, which has one tile to write and its own
// NOC otherwise idle: this reader takes the first H0 of each run.
//
// Compile-time args: 0 PAGE, 1 SPLIT_NUM, 2 SPLIT_DEN, 3.. TensorAccessorArgs for x
// Runtime args: 0 x_addr, 1 work_lo, 2 work_hi, 3 run_lo, 4 run_len

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t PAGE = get_compile_time_arg_val(0);
    constexpr uint32_t SPLIT_NUM = get_compile_time_arg_val(1);
    constexpr uint32_t SPLIT_DEN = get_compile_time_arg_val(2);
    constexpr uint32_t cb_x = 0;

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t work_lo = get_arg_val<uint32_t>(1);
    const uint32_t work_hi = get_arg_val<uint32_t>(2);
    const uint32_t run_lo = get_arg_val<uint32_t>(3);
    const uint32_t run_len = get_arg_val<uint32_t>(4);

    constexpr auto x_ta = TensorAccessorArgs<3>();
    const auto x_acc = TensorAccessor(x_ta, x_addr);

    if (work_lo >= work_hi) {
        return;
    }
    const uint32_t h0 = (run_len * SPLIT_NUM) / SPLIT_DEN;
    if (h0 == 0) {
        return;
    }
    cb_reserve_back(cb_x, h0);
    const uint32_t b = get_write_ptr(cb_x);
    for (uint32_t j = 0; j < h0; ++j) {
        noc_async_read_page(run_lo + j, x_acc, b + j * PAGE);
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, h0);
}
