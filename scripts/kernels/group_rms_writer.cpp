// group_rms_writer.cpp -- the rest of the run, and this core's partial.
//
// With FOLD, the two reduction passes are one launch. Each group's `PARTS` cores
// compute a partial; the one at part 0 is the gatherer. The others write their
// partial straight into the gatherer's fold buffer -- same circular buffer, same
// L1 offset on every core -- and signal. The gatherer waits for the count, pushes
// the whole buffer to its own compute, and writes the finished scale.
//
// Splitting the reduction is not optional: one core over a whole 80-tile group is
// 240 SFPU tile operations and measured 31 us. What *was* optional is the second
// launch to fold the partials, and this removes it.
//
// Compile-time args: 0 PAGE, 1 SPLIT_NUM, 2 SPLIT_DEN, 3 FOLD, 4 PARTS,
//                    5.. TensorAccessorArgs for out, x
// Runtime args: 0 out_addr, 1 x_addr, 2 work_lo, 3 work_hi, 4 run_lo, 5 run_len,
//               6 part, 7 gather_x, 8 gather_y, 9 scale_page

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t PAGE = get_compile_time_arg_val(0);
    constexpr uint32_t SPLIT_NUM = get_compile_time_arg_val(1);
    constexpr uint32_t SPLIT_DEN = get_compile_time_arg_val(2);
    constexpr uint32_t FOLD = get_compile_time_arg_val(3);
    constexpr uint32_t PARTS = get_compile_time_arg_val(4);
    constexpr uint32_t cb_x2 = 1, cb_out = 2, cb_fold = 3;

    const uint32_t o_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_arg_val<uint32_t>(1);
    const uint32_t work_lo = get_arg_val<uint32_t>(2);
    const uint32_t work_hi = get_arg_val<uint32_t>(3);
    const uint32_t run_lo = get_arg_val<uint32_t>(4);
    const uint32_t run_len = get_arg_val<uint32_t>(5);

    constexpr auto o_ta = TensorAccessorArgs<5>();
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
    if constexpr (!FOLD) {
        noc_async_write_page(work_lo, o_acc, get_read_ptr(cb_out));
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
        return;
    }

    const uint32_t part = get_arg_val<uint32_t>(6);
    const uint32_t gx = get_arg_val<uint32_t>(7);
    const uint32_t gy = get_arg_val<uint32_t>(8);
    const uint32_t scale_page = get_arg_val<uint32_t>(9);
    const uint32_t fold_base = get_write_ptr(cb_fold);
    volatile tt_l1_ptr uint32_t* sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));

    // Everyone's partial lands in the gatherer's fold buffer at its own slot.
    noc_async_write(get_read_ptr(cb_out),
                    get_noc_addr(gx, gy, fold_base + part * PAGE), PAGE);
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
    if (part != 0) {
        noc_semaphore_inc(get_noc_addr(gx, gy, (uint32_t)get_semaphore(0)), 1);
        return;
    }

    noc_semaphore_wait(sem, PARTS - 1);
    noc_semaphore_set(sem, 0);
    cb_reserve_back(cb_fold, PARTS);
    cb_push_back(cb_fold, PARTS);       // the tiles are already in place

    cb_wait_front(cb_out, 1);           // the compute's second window
    noc_async_write_page(scale_page, o_acc, get_read_ptr(cb_out));
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
}
