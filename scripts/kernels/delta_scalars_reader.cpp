// delta_scalars_reader.cpp -- three tiles for the DeltaNet decay/beta chain.
//
// The chain it feeds is nine ttnn calls a layer on tensors of one tile:
//   a  = both_ab[.., :H]        b = both_ab[.., H:]
//   g_exp = exp(A * softplus(a + dt))       reshaped to [H, 1, 1, 1]
//   beta  = sigmoid(b)                      reshaped to [H, 1, 1, 1]
// 324 launches a token to turn 24 numbers into 24 numbers, and every one of them
// pays a full-grid dispatch. This whole file's work is three page reads.
//
// Compile-time args: 0.. TensorAccessorArgs for both_ab, dt, a_decay
// Runtime args: 0 ab_addr, 1 dt_addr, 2 a_addr

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_ab = 0, cb_dt = 1, cb_ad = 2;

    constexpr auto ab_ta = TensorAccessorArgs<0>();
    const auto ab_acc = TensorAccessor(ab_ta, get_arg_val<uint32_t>(0));
    constexpr auto dt_ta = TensorAccessorArgs<ab_ta.next_compile_time_args_offset()>();
    const auto dt_acc = TensorAccessor(dt_ta, get_arg_val<uint32_t>(1));
    constexpr auto ad_ta = TensorAccessorArgs<dt_ta.next_compile_time_args_offset()>();
    const auto ad_acc = TensorAccessor(ad_ta, get_arg_val<uint32_t>(2));

    cb_reserve_back(cb_ab, 1);
    cb_reserve_back(cb_dt, 1);
    cb_reserve_back(cb_ad, 1);
    noc_async_read_page(0, ab_acc, get_write_ptr(cb_ab));
    noc_async_read_page(0, dt_acc, get_write_ptr(cb_dt));
    noc_async_read_page(0, ad_acc, get_write_ptr(cb_ad));
    noc_async_read_barrier();
    cb_push_back(cb_ab, 1);
    cb_push_back(cb_dt, 1);
    cb_push_back(cb_ad, 1);
}
