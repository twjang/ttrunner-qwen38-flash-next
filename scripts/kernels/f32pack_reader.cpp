// f32pack_reader.cpp -- can a generic_op kernel write a float32 output here?
//
// INVARIANT 104 says packing into a Float32 circular buffer hangs the packer,
// bisected on the k-split's partials. That was measured only with
// `fp32_dest_acc_en=True`, and it blocks every future kernel that has to write
// a float32 tensor -- the DeltaNet recurrent state, for one, which is float32
// and whose update is the next fusion worth having.
//
// This is the smallest thing that separates the two: copy N tiles through the
// compute kernel and write them back, with the destination format and the
// accumulate mode as knobs.
//
// Compile-time args: 0 PAGE, 1.. accessors in, out
// Runtime args: 0 in, 1 out, 2 lo, 3 len

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t PAGE = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in = 0;
    const uint32_t i_addr = get_arg_val<uint32_t>(0);
    const uint32_t lo = get_arg_val<uint32_t>(2);
    const uint32_t len = get_arg_val<uint32_t>(3);
    constexpr auto i_ta = TensorAccessorArgs<1>();
    const auto i_acc = TensorAccessor(i_ta, i_addr);
    if (len == 0) return;
    for (uint32_t j = 0; j < len; ++j) {
        cb_reserve_back(cb_in, 1);
        noc_async_read_page(lo + j, i_acc, get_write_ptr(cb_in));
        noc_async_read_barrier();
        cb_push_back(cb_in, 1);
    }
}
