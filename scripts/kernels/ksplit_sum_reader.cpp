// ksplit_sum_reader.cpp -- stream the k-split's G partials for one output tile.
//
// `ksplit_linear` splits a matmul's reduction across cores and leaves G partial
// results in a [1, G, M, N] tensor; `ttnn.sum(dim=1)` then adds them. That sum is
// 193 calls a token at **6.51 us** apiece -- a full-grid launch to reduce, at
// most, a hundred and ten tiles down to ten. `dispatch_floor.py` prices ten
// cores at 2.46 us.
//
// Work item = one output tile. Partial j of output tile t is at page
// j * STRIDE + t, where STRIDE is the tiles in one partial (M_t * N_t).
//
// All G reads are issued before one barrier. Issuing them one at a time with a
// barrier each made the kernel *slower than the op*: G serialised DRAM round
// trips a tile, ~6.7 us against ttnn.sum's 6.64, and 7.02 against 3.37 on the
// narrow shapes.
//
// Compile-time args: 0 GROUPS, 1 STRIDE, 2 PAGE_BYTES,
//                    3.. TensorAccessorArgs for the partials
// Runtime args: 0 src_addr, 1 work_lo, 2 work_hi

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t GROUPS = get_compile_time_arg_val(0);
    constexpr uint32_t STRIDE = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in = 0;

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t work_lo = get_arg_val<uint32_t>(1);
    const uint32_t work_hi = get_arg_val<uint32_t>(2);

    constexpr uint32_t PAGE = get_compile_time_arg_val(2);
    constexpr auto s_ta = TensorAccessorArgs<3>();
    const auto s_acc = TensorAccessor(s_ta, src_addr);

    for (uint32_t w = work_lo; w < work_hi; ++w) {
        cb_reserve_back(cb_in, GROUPS);
        const uint32_t base = get_write_ptr(cb_in);
        for (uint32_t g = 0; g < GROUPS; ++g) {
            noc_async_read_page(g * STRIDE + w, s_acc, base + g * PAGE);
        }
        noc_async_read_barrier();
        cb_push_back(cb_in, GROUPS);
    }
}
