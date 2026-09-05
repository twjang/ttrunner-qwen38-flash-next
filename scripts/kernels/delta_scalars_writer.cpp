// delta_scalars_writer.cpp -- scatter H values per output into H one-value tiles.
//
// `g_exp` and `beta` are [H, 1, 1, 1]: H pages, each a tile whose only live
// element is (0, 0). That is what the two `ttnn.reshape` calls this replaces
// produce, and what the broadcast multiplies in `decode_step` read.
//
// Only the first 64 bytes of each destination page are written. The rest is
// never touched, and the buffers are allocated zeroed and reused for the life of
// the process, so the pad stays zero -- the same contract `_ROPE_OUT` relies on.
// 64 rather than one element because a Blackhole DRAM transfer needs
// (local & 63) == (noc & 63), and the page offset is 0.
//
// Element (0, c) of a tile is at face (c / 16), byte (c % 16) * ELEM within it.
//
// Compile-time args: 0 H, 1 ELEM_BYTES, 2 FACE_BYTES, 3 TILE_BYTES,
//                    4.. TensorAccessorArgs for g_exp, then beta
// Runtime args: 0 g_addr, 1 beta_addr

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t ELEM = get_compile_time_arg_val(1);
    constexpr uint32_t FACE = get_compile_time_arg_val(2);
    constexpr uint32_t TILE_BYTES = get_compile_time_arg_val(3);
    constexpr uint32_t CHUNK = 64;

    constexpr uint32_t cb_out = 3, cb_scratch = 4;

    constexpr auto g_ta = TensorAccessorArgs<4>();
    const auto g_acc = TensorAccessor(g_ta, get_arg_val<uint32_t>(0));
    constexpr auto b_ta = TensorAccessorArgs<g_ta.next_compile_time_args_offset()>();
    const auto b_acc = TensorAccessor(b_ta, get_arg_val<uint32_t>(1));

    // 2H aligned slots, so every value has its own source and one barrier covers
    // every write.
    const uint32_t scratch = (get_write_ptr(cb_scratch) + 63u) & ~63u;
    volatile tt_l1_ptr uint8_t* s8 = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(scratch);
    for (uint32_t i = 0; i < 2 * H * CHUNK; ++i) {
        s8[i] = 0;
    }

    // The decay tile first, then the beta tile -- the order the compute pushed
    // them, so they are consecutive pages of the same circular buffer.
    cb_wait_front(cb_out, 2);
    const uint32_t g_tile = get_read_ptr(cb_out);
    const uint32_t b_tile = g_tile + TILE_BYTES;

    for (uint32_t h = 0; h < H; ++h) {
        const uint32_t gc = h;                 // g lives in the first half
        const uint32_t bc = H + h;             // beta in the second
        volatile tt_l1_ptr uint8_t* gs = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(
            g_tile + (gc / 16) * FACE + (gc % 16) * ELEM);
        volatile tt_l1_ptr uint8_t* bs = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(
            b_tile + (bc / 16) * FACE + (bc % 16) * ELEM);
        for (uint32_t e = 0; e < ELEM; ++e) {
            s8[h * CHUNK + e] = gs[e];
            s8[(H + h) * CHUNK + e] = bs[e];
        }
        noc_async_write(scratch + h * CHUNK, g_acc.get_noc_addr(h, 0), CHUNK);
        noc_async_write(scratch + (H + h) * CHUNK, b_acc.get_noc_addr(h, 0), CHUNK);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, 2);
}
