// qkv_heads_reader.cpp -- split the post-conv qkv stream into per-head tiles.
//
// The chain this replaces is ten ttnn calls a layer -- three slices, three
// reshapes, two `rms_norm`s and two scales -- costing **42.23 us** to turn one
// [1, 1, 1, 3*Dk] row into three [H, 1, 1, hd] stacks. None of it moves data
// anywhere interesting: at M = 1 the slice boundaries are tile-aligned and the
// reshape is the identity page mapping, so head h's tiles are already exactly
// pages TPH*h .. TPH*h+TPH-1 of their section.
//
// So the split is a page copy, and the only arithmetic is the two l2 norms.
//
//   q head h, tile j   <-  qkv page QBASE + TPH*h + j
//   k head h, tile j   <-  qkv page KBASE + TPH*h + j
//   v head h, tile j   <-  qkv page VBASE + TPH*h + j
//
// `v` is not normalised, so the reader copies it straight through rather than
// sending it round the compute kernel.
//
// **M = 1 only.** With more than one row the destination separates (b, h) into
// its own page while the source keeps all rows of a column in one, and the copy
// stops being a copy.
//
// Compile-time args: 0 QBASE, 1 KBASE, 2 VBASE, 3 TPH,
//                    4 PAGE_BYTES, 5.. TensorAccessorArgs for qkv, then v_out
// Runtime args: 0 qkv_addr, 1 v_addr, 2 head_lo, 3 head_hi

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t QBASE = get_compile_time_arg_val(0);
    constexpr uint32_t KBASE = get_compile_time_arg_val(1);
    constexpr uint32_t VBASE = get_compile_time_arg_val(2);
    constexpr uint32_t TPH = get_compile_time_arg_val(3);
    constexpr uint32_t PAGE_BYTES = get_compile_time_arg_val(4);

    constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 4;

    const uint32_t head_lo = get_arg_val<uint32_t>(2);
    const uint32_t head_hi = get_arg_val<uint32_t>(3);

    constexpr auto s_ta = TensorAccessorArgs<5>();
    const auto s_acc = TensorAccessor(s_ta, get_arg_val<uint32_t>(0));
    constexpr auto v_ta = TensorAccessorArgs<s_ta.next_compile_time_args_offset()>();
    const auto v_acc = TensorAccessor(v_ta, get_arg_val<uint32_t>(1));

    for (uint32_t h = head_lo; h < head_hi; ++h) {
        const uint32_t off = TPH * h;

        cb_reserve_back(cb_q, TPH);
        const uint32_t qb = get_write_ptr(cb_q);
        cb_reserve_back(cb_k, TPH);
        const uint32_t kb = get_write_ptr(cb_k);
        for (uint32_t j = 0; j < TPH; ++j) {
            noc_async_read_page(QBASE + off + j, s_acc, qb + j * PAGE_BYTES);
            noc_async_read_page(KBASE + off + j, s_acc, kb + j * PAGE_BYTES);
        }
        noc_async_read_barrier();
        cb_push_back(cb_q, TPH);
        cb_push_back(cb_k, TPH);

        // v straight through, one tile at a time so the scratch stays small.
        for (uint32_t j = 0; j < TPH; ++j) {
            const uint32_t vb = get_write_ptr(cb_v);
            noc_async_read_page(VBASE + off + j, s_acc, vb);
            noc_async_read_barrier();
            noc_async_write_page(off + j, v_acc, vb);
            noc_async_write_barrier();
        }
    }
}
