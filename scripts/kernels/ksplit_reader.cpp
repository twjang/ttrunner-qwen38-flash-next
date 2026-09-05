// ksplit_reader.cpp -- feed a GEMV whose reduction is split across cores.
//
// `ttnn.linear` at M=1 runs at 25-33 % of bandwidth when its output is narrow,
// because N sets how many output tiles there are and therefore how many cores
// get work: [2560, 320] is ten tiles, so ten cores of a hundred and thirty do
// the whole thing. That is invariant 38, and it is ~43 ms of the step.
//
// The fix is to give the idle cores a slice of the *reduction* instead. Core
// (g, nt) accumulates only K-tiles [kt_lo, kt_hi) for output column nt and
// writes a partial; the host sums the G partials with one `ttnn.sum`. No
// cross-core semaphore, no second kernel pass.
//
// The evidence that this is where the time is: `hc_up` [320, 10240] does four
// times the parameters of `hc_down` [2560, 320] in half the time -- 14.45 us
// against 31.50 -- purely because its output is wide.
//
// Compile-time args:
//   0: KT        (K / 32, tiles down the reduction)
//   1: NT        (N / 32, tiles across the output)
//   2..: TensorAccessorArgs for the activation, then the weight
//
// Per-core runtime args: 0 a_addr, 1 w_addr, 2 kt_lo, 3 kt_hi, 4 nt

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t NT = get_compile_time_arg_val(1);
    constexpr uint32_t A_PAGE = get_compile_time_arg_val(2);
    constexpr uint32_t W_PAGE = get_compile_time_arg_val(3);
    constexpr uint32_t cb_a = 0;
    constexpr uint32_t cb_b = 1;

    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t kt_lo = get_arg_val<uint32_t>(2);
    const uint32_t kt_hi = get_arg_val<uint32_t>(3);
    const uint32_t nt = get_arg_val<uint32_t>(4);

    constexpr auto a_ta = TensorAccessorArgs<4>();
    const auto a_acc = TensorAccessor(a_ta, a_addr);
    constexpr auto w_ta = TensorAccessorArgs<a_ta.next_compile_time_args_offset()>();
    const auto w_acc = TensorAccessor(w_ta, w_addr);

    // All of this core's reads against **one** barrier. Issued one k-tile at a
    // time with a barrier each -- as this was written -- every core pays a
    // serialised DRAM round trip per tile, and the split loses to the plain
    // matmul it was built to beat. The same fix was worth 1.6x on the k-split's
    // own reduction kernel.
    const uint32_t n = kt_hi - kt_lo;
    if (n == 0) {
        return;
    }
    cb_reserve_back(cb_a, n);
    cb_reserve_back(cb_b, n);
    const uint32_t ab = get_write_ptr(cb_a);
    const uint32_t bb = get_write_ptr(cb_b);
    for (uint32_t i = 0; i < n; ++i) {
        noc_async_read_page(kt_lo + i, a_acc, ab + i * A_PAGE);
        noc_async_read_page((kt_lo + i) * NT + nt, w_acc, bb + i * W_PAGE);
    }
    noc_async_read_barrier();
    cb_push_back(cb_a, n);
    cb_push_back(cb_b, n);
}
