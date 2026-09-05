// group_rms_writer.cpp -- the rest of the run, and the one tile this core makes.
//
// Compile-time args: 0 PAGE, 1 SPLIT_NUM, 2 SPLIT_DEN,
//                    3.. TensorAccessorArgs for out, x
// Runtime args: 0 out_addr, 1 x_addr, 2 work_lo, 3 work_hi, 4 run_lo, 5 run_len

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t PAGE = get_compile_time_arg_val(0);
    constexpr uint32_t SPLIT_NUM = get_compile_time_arg_val(1);
    constexpr uint32_t SPLIT_DEN = get_compile_time_arg_val(2);
    constexpr uint32_t cb_x2 = 1, cb_out = 2;

    const uint32_t o_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_arg_val<uint32_t>(1);
    const uint32_t work_lo = get_arg_val<uint32_t>(2);
    const uint32_t work_hi = get_arg_val<uint32_t>(3);
    const uint32_t run_lo = get_arg_val<uint32_t>(4);
    const uint32_t run_len = get_arg_val<uint32_t>(5);

    constexpr auto o_ta = TensorAccessorArgs<3>();
    const auto o_acc = TensorAccessor(o_ta, o_addr);
    constexpr auto x_ta = TensorAccessorArgs<o_ta.next_compile_time_args_offset()>();
    const auto x_acc = TensorAccessor(x_ta, x_addr);

    if (work_lo >= work_hi) {
        return;
    }
    const uint32_t h0 = (run_len * SPLIT_NUM) / SPLIT_DEN;
    const uint32_t h1 = run_len - h0;
    if (h1 > 0) {
        cb_reserve_back(cb_x2, h1);
        const uint32_t b = get_write_ptr(cb_x2);
        for (uint32_t j = 0; j < h1; ++j) {
            noc_async_read_page(run_lo + h0 + j, x_acc, b + j * PAGE);
        }
        noc_async_read_barrier();
        cb_push_back(cb_x2, h1);
    }
    cb_wait_front(cb_out, 1);
    noc_async_write_page(work_lo, o_acc, get_read_ptr(cb_out));
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
}
