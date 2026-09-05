// delta_out_reader.cpp -- one head's operands for the delta rule's tail.
//
// After the two state matmuls, `decode_step` runs six ops on [BH,1,1,Dv] tensors:
//
//     delta = (v - predicted) * beta
//     qk    = sum(q * k)
//     out   = q_decayed + qk * delta
//
// a subtract, two multiplies, a reduction and two more multiplies, 36 times a
// token. The cumulative ablation puts the whole of `decode_step` at 1.99 ms and
// these are most of its launches; invariant 91 says a chain only pays when it is
// cut deep, and this is six of them.
//
// Every operand is the same [BH,1,1,Dv] shape and one core takes one head, so
// the addressing is head h tile j -> page TPH*h + j everywhere. `beta` is one
// scalar a head, read as page h and used through `mul_tiles_bcast_scalar`,
// which looks only at element (0, 0).
//
// Compile-time args: 0 TPH, 1 PAGE,
//                    2.. TensorAccessorArgs for v, predicted, beta, q, k, q_decayed
// Runtime args: 0 v, 1 pred, 2 beta, 3 q, 4 k, 5 qdec, 6 head_lo, 7 head_hi

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t TPH = get_compile_time_arg_val(0);
    constexpr uint32_t PAGE = get_compile_time_arg_val(1);
    constexpr uint32_t cb_v = 0, cb_p = 1, cb_b = 2, cb_q = 3, cb_k = 4, cb_d = 5;

    const uint32_t head_lo = get_arg_val<uint32_t>(6);
    const uint32_t head_hi = get_arg_val<uint32_t>(7);

    constexpr auto v_ta = TensorAccessorArgs<2>();
    const auto v_acc = TensorAccessor(v_ta, get_arg_val<uint32_t>(0));
    constexpr auto p_ta = TensorAccessorArgs<v_ta.next_compile_time_args_offset()>();
    const auto p_acc = TensorAccessor(p_ta, get_arg_val<uint32_t>(1));
    constexpr auto b_ta = TensorAccessorArgs<p_ta.next_compile_time_args_offset()>();
    const auto b_acc = TensorAccessor(b_ta, get_arg_val<uint32_t>(2));
    constexpr auto q_ta = TensorAccessorArgs<b_ta.next_compile_time_args_offset()>();
    const auto q_acc = TensorAccessor(q_ta, get_arg_val<uint32_t>(3));
    constexpr auto k_ta = TensorAccessorArgs<q_ta.next_compile_time_args_offset()>();
    const auto k_acc = TensorAccessor(k_ta, get_arg_val<uint32_t>(4));
    constexpr auto d_ta = TensorAccessorArgs<k_ta.next_compile_time_args_offset()>();
    const auto d_acc = TensorAccessor(d_ta, get_arg_val<uint32_t>(5));

    for (uint32_t h = head_lo; h < head_hi; ++h) {
        const uint32_t off = TPH * h;
        cb_reserve_back(cb_v, TPH);
        cb_reserve_back(cb_p, TPH);
        cb_reserve_back(cb_q, TPH);
        cb_reserve_back(cb_k, TPH);
        cb_reserve_back(cb_d, TPH);
        cb_reserve_back(cb_b, 1);
        const uint32_t vb = get_write_ptr(cb_v), pb = get_write_ptr(cb_p);
        const uint32_t qb = get_write_ptr(cb_q), kb = get_write_ptr(cb_k);
        const uint32_t db = get_write_ptr(cb_d);
        for (uint32_t j = 0; j < TPH; ++j) {
            noc_async_read_page(off + j, v_acc, vb + j * PAGE);
            noc_async_read_page(off + j, p_acc, pb + j * PAGE);
            noc_async_read_page(off + j, q_acc, qb + j * PAGE);
            noc_async_read_page(off + j, k_acc, kb + j * PAGE);
            noc_async_read_page(off + j, d_acc, db + j * PAGE);
        }
        noc_async_read_page(h, b_acc, get_write_ptr(cb_b));
        noc_async_read_barrier();
        cb_push_back(cb_v, TPH);
        cb_push_back(cb_p, TPH);
        cb_push_back(cb_q, TPH);
        cb_push_back(cb_k, TPH);
        cb_push_back(cb_d, TPH);
        cb_push_back(cb_b, 1);
    }
}
