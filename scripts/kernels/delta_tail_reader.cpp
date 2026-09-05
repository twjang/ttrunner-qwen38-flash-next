// delta_tail_reader.cpp -- one head's recurrent output, gate and norm weight.
//
// The chain this replaces is six ttnn calls a layer: two reshapes to bring the
// head stack and the gate into [1, 1, H, hd], an `rms_norm` with a weight, a
// sigmoid, a multiply, and a reshape back out -- **28.73 us**, 1.03 ms a token.
//
// The reshapes exist only because `ttnn.rms_norm` reduces over the last axis, so
// the heads have to be rows first. A kernel that does its own reduction does not
// need them, and then every mapping is the identity:
//
//   head h, tile j:  out page h*TPH + j,  z page h*TPH + j,
//                    weight page j,       gated page h*TPH + j
//
// Compile-time args: 0 TPH, 1 PAGE_BYTES,
//                    2.. TensorAccessorArgs for out, z, weight
// Runtime args: 0 out_addr, 1 z_addr, 2 w_addr, 3 head_lo, 4 head_hi

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t TPH = get_compile_time_arg_val(0);
    constexpr uint32_t PAGE_BYTES = get_compile_time_arg_val(1);

    constexpr uint32_t cb_o = 0, cb_z = 1, cb_w = 2;

    const uint32_t head_lo = get_arg_val<uint32_t>(3);
    const uint32_t head_hi = get_arg_val<uint32_t>(4);

    constexpr auto o_ta = TensorAccessorArgs<2>();
    const auto o_acc = TensorAccessor(o_ta, get_arg_val<uint32_t>(0));
    constexpr auto z_ta = TensorAccessorArgs<o_ta.next_compile_time_args_offset()>();
    const auto z_acc = TensorAccessor(z_ta, get_arg_val<uint32_t>(1));
    constexpr auto w_ta = TensorAccessorArgs<z_ta.next_compile_time_args_offset()>();
    const auto w_acc = TensorAccessor(w_ta, get_arg_val<uint32_t>(2));

    for (uint32_t h = head_lo; h < head_hi; ++h) {
        const uint32_t off = TPH * h;
        cb_reserve_back(cb_o, TPH);
        const uint32_t ob = get_write_ptr(cb_o);
        cb_reserve_back(cb_z, TPH);
        const uint32_t zb = get_write_ptr(cb_z);
        cb_reserve_back(cb_w, TPH);
        const uint32_t wb = get_write_ptr(cb_w);
        for (uint32_t j = 0; j < TPH; ++j) {
            noc_async_read_page(off + j, o_acc, ob + j * PAGE_BYTES);
            noc_async_read_page(off + j, z_acc, zb + j * PAGE_BYTES);
            noc_async_read_page(j, w_acc, wb + j * PAGE_BYTES);
        }
        noc_async_read_barrier();
        cb_push_back(cb_o, TPH);
        cb_push_back(cb_z, TPH);
        cb_push_back(cb_w, TPH);
    }
}
