// recur_reader.cpp -- one (head, output-tile) column's operands for the whole
// delta-rule recurrence.
//
// `decode_step` is 2.95 ms a token (handoff 45.23), 82 us a layer, and its seven
// launches each cost ~12 us against 4 us of state bytes: the cost is writing
// `decayed` and `update` out and reading them back, not the launches themselves
// (invariant 101 prices a marginal launch at 0.8 us). One kernel that reads the
// state once and writes it once removes all of it.
//
// Core (h, j) owns output column-tile j of head h and needs, for i in 0..DKT-1:
//
//     state[h, i, j]   page h*DKT*DVT + i*DVT + j
//     q[h, i], k[h, i] page h*DKT + i
//     kt[h, i]         page h*DKT + i   -- k transposed, [Dk, 1], so Wt is 1
//                                          and the tile index is the same
//     v[h, j]          page h*DVT + j
//     g_exp[h], beta[h] page h
//
// Everything the core computes -- predicted, q_decayed, delta, out and the new
// state column -- comes from those alone, so no core talks to any other and the
// program carries no semaphores. That also keeps it clear of 45.10's idle-core
// defect by construction.
//
// Compile-time args: 0 DKT, 1 DVT, 2 SPAGE, 3 VPAGE,
//                    4.. TensorAccessorArgs for state, q, k, v, g, b
// Runtime args: 0 state, 1 q, 2 k, 3 v, 4 g, 5 b, 6 head, 7 j, 8 active, 9 kt

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t DKT = get_compile_time_arg_val(0);
    constexpr uint32_t DVT = get_compile_time_arg_val(1);
    constexpr uint32_t SPAGE = get_compile_time_arg_val(2);
    constexpr uint32_t VPAGE = get_compile_time_arg_val(3);
    constexpr uint32_t cb_s = 0, cb_q = 1, cb_k = 2, cb_v = 3, cb_g = 4, cb_b = 5,
                       cb_kt = 6;

    const uint32_t head = get_arg_val<uint32_t>(6);
    const uint32_t j = get_arg_val<uint32_t>(7);
    const uint32_t active = get_arg_val<uint32_t>(8);
    if (active == 0) {
        return;                      // a core this plan does not use (45.10)
    }

    constexpr auto s_ta = TensorAccessorArgs<4>();
    const auto s_acc = TensorAccessor(s_ta, get_arg_val<uint32_t>(0));
    constexpr auto q_ta = TensorAccessorArgs<s_ta.next_compile_time_args_offset()>();
    const auto q_acc = TensorAccessor(q_ta, get_arg_val<uint32_t>(1));
    constexpr auto k_ta = TensorAccessorArgs<q_ta.next_compile_time_args_offset()>();
    const auto k_acc = TensorAccessor(k_ta, get_arg_val<uint32_t>(2));
    constexpr auto v_ta = TensorAccessorArgs<k_ta.next_compile_time_args_offset()>();
    const auto v_acc = TensorAccessor(v_ta, get_arg_val<uint32_t>(3));
    constexpr auto g_ta = TensorAccessorArgs<v_ta.next_compile_time_args_offset()>();
    const auto g_acc = TensorAccessor(g_ta, get_arg_val<uint32_t>(4));
    constexpr auto b_ta = TensorAccessorArgs<g_ta.next_compile_time_args_offset()>();
    const auto b_acc = TensorAccessor(b_ta, get_arg_val<uint32_t>(5));
    constexpr auto kt_ta = TensorAccessorArgs<b_ta.next_compile_time_args_offset()>();
    const auto kt_acc = TensorAccessor(kt_ta, get_arg_val<uint32_t>(9));

    // The scalars first: the compute kernel wants them resident for the whole
    // pass and they are one tile each.
    cb_reserve_back(cb_g, 1);
    noc_async_read_page(head, g_acc, get_write_ptr(cb_g));
    cb_reserve_back(cb_b, 1);
    noc_async_read_page(head, b_acc, get_write_ptr(cb_b));
    cb_reserve_back(cb_v, 1);
    noc_async_read_page(head * DVT + j, v_acc, get_write_ptr(cb_v));

    cb_reserve_back(cb_q, DKT);
    cb_reserve_back(cb_k, DKT);
    cb_reserve_back(cb_kt, DKT);
    const uint32_t qb = get_write_ptr(cb_q), kb = get_write_ptr(cb_k);
    const uint32_t tb = get_write_ptr(cb_kt);
    for (uint32_t i = 0; i < DKT; ++i) {
        noc_async_read_page(head * DKT + i, q_acc, qb + i * VPAGE);
        noc_async_read_page(head * DKT + i, k_acc, kb + i * VPAGE);
        // kt is [Dk, 1]: Wt is 1, so tile (i, 0) is page head*DKT + i, the same
        // index as k's. Leaving this read out is what deadlocked the state
        // write-back -- the compute kernel waited on a buffer nothing filled,
        // which is why stages 0 through 5 ran and only 6 hung.
        noc_async_read_page(head * DKT + i, kt_acc, tb + i * SPAGE);
    }

    // The state column: DKT tiles, read once and never again.
    cb_reserve_back(cb_s, DKT);
    const uint32_t sb = get_write_ptr(cb_s);
    for (uint32_t i = 0; i < DKT; ++i) {
        noc_async_read_page(head * DKT * DVT + i * DVT + j, s_acc, sb + i * SPAGE);
    }
    noc_async_read_barrier();

    cb_push_back(cb_g, 1);
    cb_push_back(cb_b, 1);
    cb_push_back(cb_v, 1);
    cb_push_back(cb_q, DKT);
    cb_push_back(cb_k, DKT);
    cb_push_back(cb_kt, DKT);
    cb_push_back(cb_s, DKT);
}
