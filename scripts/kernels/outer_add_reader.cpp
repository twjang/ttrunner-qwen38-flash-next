// outer_add_reader.cpp -- state = decayed + kt (x) delta, in one launch.
//
// The DeltaNet state update is two ops today:
//
//     update = ttnn.multiply(kt, delta)          # [BH,1,Dk,1] x [BH,1,1,Dv]
//     ttnn.add(decayed, update, output_tensor=state)
//
// `update` is a full state-sized tensor -- 786 KB at batch 1 -- written once and
// read once, and nothing else ever looks at it. Thirty-six layers a token is
// **57 MB of DRAM that exists only to carry a value between two launches**,
// about 0.15 ms at 388 GB/s. This is invariant 101's own prescription: fuse to
// stop moving the same bytes twice, not to save a dispatch.
//
// No cross-core anything: each core owns a run of output tiles and reads exactly
// the three tiles each needs. Nothing here can hang the card.
//
// The outer product needs both a column broadcast and a row broadcast, and the
// FPU does one at a time, so a tile of ones turns kt's column into a full tile
// first. It is one page, read once per core, and the host caches it.
//
// Tile indexing. The state is [BH, 1, DK, DV] with DKT x DVT tiles a head, so
// output tile t = h*DKT*DVT + a*DVT + b. Its operands are:
//     decayed  the same page t
//     kt       [BH, 1, DK, 1]  -> page h*DKT + a   (one column, padded to a tile)
//     delta    [BH, 1, 1, DV]  -> page h*DVT + b
//
// Compile-time args: 0 PAGE, 1 DKT, 2 DVT, 3.. accessors decayed, kt, delta, ones
// Runtime args: 0 decayed, 1 kt, 2 delta, 3 ones, 4 tile_lo, 5 tile_len

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t PAGE = get_compile_time_arg_val(0);
    constexpr uint32_t DKT = get_compile_time_arg_val(1);
    constexpr uint32_t DVT = get_compile_time_arg_val(2);
    constexpr uint32_t cb_d = 0, cb_k = 1, cb_v = 2, cb_1 = 3;

    const uint32_t d_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t o_addr = get_arg_val<uint32_t>(3);
    const uint32_t lo = get_arg_val<uint32_t>(4);
    const uint32_t len = get_arg_val<uint32_t>(5);

    constexpr auto d_ta = TensorAccessorArgs<3>();
    const auto d_acc = TensorAccessor(d_ta, d_addr);
    constexpr auto k_ta = TensorAccessorArgs<d_ta.next_compile_time_args_offset()>();
    const auto k_acc = TensorAccessor(k_ta, k_addr);
    constexpr auto v_ta = TensorAccessorArgs<k_ta.next_compile_time_args_offset()>();
    const auto v_acc = TensorAccessor(v_ta, v_addr);
    constexpr auto o_ta = TensorAccessorArgs<v_ta.next_compile_time_args_offset()>();
    const auto o_acc = TensorAccessor(o_ta, o_addr);

    if (len == 0) {
        return;
    }
    // The ones tile, once, and it stays: every output tile multiplies by it.
    cb_reserve_back(cb_1, 1);
    noc_async_read_page(0, o_acc, get_write_ptr(cb_1));
    noc_async_read_barrier();
    cb_push_back(cb_1, 1);

    constexpr uint32_t PER_HEAD = DKT * DVT;
    for (uint32_t j = 0; j < len; ++j) {
        const uint32_t t = lo + j;
        const uint32_t h = t / PER_HEAD;
        const uint32_t r = t - h * PER_HEAD;
        const uint32_t a = r / DVT;
        const uint32_t b = r - a * DVT;

        cb_reserve_back(cb_d, 1);
        cb_reserve_back(cb_k, 1);
        cb_reserve_back(cb_v, 1);
        noc_async_read_page(t, d_acc, get_write_ptr(cb_d));
        noc_async_read_page(h * DKT + a, k_acc, get_write_ptr(cb_k));
        noc_async_read_page(h * DVT + b, v_acc, get_write_ptr(cb_v));
        noc_async_read_barrier();
        cb_push_back(cb_d, 1);
        cb_push_back(cb_k, 1);
        cb_push_back(cb_v, 1);
    }
}
