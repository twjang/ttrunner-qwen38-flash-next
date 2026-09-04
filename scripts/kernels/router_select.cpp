// router_select.cpp -- the whole routing tail in one kernel: global top-k
// threshold, tie-admitting mask, normalisation, and this device's local
// selection with its gather indices.
//
// What it replaces, measured one at a time with `decode_ablation_check.py`
// against an 82.11 ms step:
//
//     ttnn.topk(probs, k=10) over 512     5.00 ms   <- the single biggest op
//     slice / ge / multiply / sum / divide 0.93
//     ttnn.mesh_partition                  0.73
//     ttnn.topk(weights_local, k_sel=10)   2.00      (inside wide_expert_ffn)
//                                          ----
//                                          8.66 ms a token, over 48 layers
//
// against one launch a layer. Nothing here needs the FPU: after a softmax every
// probability is positive, and for positive IEEE floats the bit pattern orders
// exactly as the number does, so the whole selection is unsigned integer
// compares on a scalar RISC-V core. Only the normalisation is real arithmetic --
// about a dozen adds and ten divides a row.
//
// Exactness. The chain being replaced admits ties (`probs >= threshold`, so a
// row keeps ~11.5 experts where k is 10) and normalises by the sum over all of
// them; this reproduces that rule rather than a clean top-10, because changing
// it would change the model's output. The one place the two can differ is an
// exact tie at the k_sel-th *local* position, where `ttnn.topk` and a linear
// scan may keep different members of the tie -- the same measure-zero class as
// the tie admission the chain already documents.
//
// Layout notes, all forced by TILE_LAYOUT. A tile is four 16x16 faces: face 0 is
// rows 0-15 x cols 0-15, face 1 rows 0-15 x cols 16-31. So row r < 16 of tile t
// holds columns 32t+0..15 at element offset r*16 in face 0, and columns
// 32t+16..31 at 256 + r*16. Rows 16-31 live in faces 2 and 3, at +512 and +768.
//
// The index output is written the way `expert_gather.cpp` reads it under
// IDX16=1: uint16 in the first face, columns 0..15 contiguous.
//
// Positional compile-time args:
//   0: E_TOTAL        experts over the whole mesh (512)
//   1: E_LOCAL        experts on this device (128)
//   2: TOP_K          the router's k (10) -- sets the threshold
//   3: K_SEL          gather slots to fill (10)
//   4: M              real rows (1 at decode; the tile's other rows are padding)
//   5: PROBS_BF16     1: probs are bfloat16, 0: float32
//   6: VALS_BF16      1: the weight output is bfloat16, 0: float32
//   7: PROBS_TILE_BYTES
//   8: VALS_TILE_BYTES
//   9: IDX_TILE_BYTES
//  10..: TensorAccessorArgs for probs, dev_id, vals, idx
//
// Runtime args: probs_addr, devid_addr, vals_addr, idx_addr

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t E_TOTAL = get_compile_time_arg_val(0);
constexpr uint32_t E_LOCAL = get_compile_time_arg_val(1);
constexpr uint32_t TOP_K = get_compile_time_arg_val(2);
constexpr uint32_t K_SEL = get_compile_time_arg_val(3);
constexpr uint32_t M_ROWS = get_compile_time_arg_val(4);
constexpr uint32_t PROBS_BF16 = get_compile_time_arg_val(5);
constexpr uint32_t VALS_BF16 = get_compile_time_arg_val(6);
constexpr uint32_t PROBS_TILE_BYTES = get_compile_time_arg_val(7);
constexpr uint32_t VALS_TILE_BYTES = get_compile_time_arg_val(8);
constexpr uint32_t IDX_TILE_BYTES = get_compile_time_arg_val(9);

constexpr uint32_t NT_PROBS = E_TOTAL / 32;      // tiles across the probs row

static_assert(E_TOTAL % 32 == 0, "E_TOTAL must be a whole number of tiles");
static_assert(E_LOCAL % 32 == 0, "E_LOCAL must be a whole number of tiles");
static_assert(TOP_K <= 32, "TOP_K must fit the scan buffer");
static_assert(K_SEL <= 16, "K_SEL must fit one tile face row, as the gather reads it");
static_assert(M_ROWS <= 32, "M_ROWS must fit one tile");

// bfloat16 is the top half of a float32, so widening is a shift and narrowing is
// a shift with round-to-nearest-even -- the same rounding ttnn uses.
inline uint32_t bf16_to_f32_bits(uint16_t v) { return ((uint32_t)v) << 16; }

inline uint16_t f32_bits_to_bf16(uint32_t b) {
    const uint32_t lsb = (b >> 16) & 1u;
    const uint32_t bias = 0x7fffu + lsb;
    return (uint16_t)((b + bias) >> 16);
}

inline float bits_to_f32(uint32_t b) {
    union { uint32_t u; float f; } c;
    c.u = b;
    return c.f;
}

inline uint32_t f32_to_bits(float f) {
    union { uint32_t u; float f; } c;
    c.f = f;
    return c.u;
}

// Element offset of (row, col) within a 32x32 tile laid out as four 16x16 faces.
inline uint32_t tile_off(uint32_t row, uint32_t col) {
    const uint32_t face = (row < 16 ? 0u : 2u) + (col < 16 ? 0u : 1u);
    return face * 256u + (row & 15u) * 16u + (col & 15u);
}

void kernel_main() {
    const uint32_t probs_addr = get_arg_val<uint32_t>(0);
    const uint32_t devid_addr = get_arg_val<uint32_t>(1);
    const uint32_t vals_addr = get_arg_val<uint32_t>(2);
    const uint32_t idx_addr = get_arg_val<uint32_t>(3);

    constexpr auto p_ta = TensorAccessorArgs<10>();
    const auto p_acc = TensorAccessor(p_ta, probs_addr);
    constexpr auto d_ta = TensorAccessorArgs<p_ta.next_compile_time_args_offset()>();
    const auto d_acc = TensorAccessor(d_ta, devid_addr);
    constexpr auto v_ta = TensorAccessorArgs<d_ta.next_compile_time_args_offset()>();
    const auto v_acc = TensorAccessor(v_ta, vals_addr);
    constexpr auto i_ta = TensorAccessorArgs<v_ta.next_compile_time_args_offset()>();
    const auto i_acc = TensorAccessor(i_ta, idx_addr);

    // Sole owner of the CBs, so their write pointers are stable L1 scratch.
    // Both landing addresses are forced to 64 B: a Blackhole DRAM transfer
    // requires (local & 63) == (noc & 63), and a CB base is only L1-aligned.
    const uint32_t probs_l1 = (get_write_ptr(0) + 63u) & ~63u;
    const uint32_t misc_l1 = (get_write_ptr(1) + 63u) & ~63u;
    const uint32_t vals_l1 = (misc_l1 + VALS_TILE_BYTES + 63u) & ~63u;
    const uint32_t idx_l1 = (vals_l1 + VALS_TILE_BYTES + 63u) & ~63u;

    // -- this device's expert window ----------------------------------------
    noc_async_read_page(0, d_acc, misc_l1);
    // -- the whole probability row, all NT_PROBS tiles ------------------------
    for (uint32_t t = 0; t < NT_PROBS; ++t) {
        noc_async_read_page(t, p_acc, probs_l1 + t * PROBS_TILE_BYTES);
    }
    noc_async_read_barrier();

    const uint32_t dev = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(misc_l1);
    const uint32_t e_lo = dev * E_LOCAL;
    const uint32_t e_hi = e_lo + E_LOCAL;

    volatile tt_l1_ptr uint16_t* p16 =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(probs_l1);
    volatile tt_l1_ptr uint32_t* p32 =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(probs_l1);
    volatile tt_l1_ptr uint16_t* v16 =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(vals_l1);
    volatile tt_l1_ptr uint32_t* v32 =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(vals_l1);
    volatile tt_l1_ptr uint16_t* o16 =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(idx_l1);

    for (uint32_t b = 0; b < VALS_TILE_BYTES / 4; ++b) {
        v32[b] = 0;
    }
    for (uint32_t b = 0; b < IDX_TILE_BYTES / 2; ++b) {
        o16[b] = 0;
    }

    for (uint32_t row = 0; row < M_ROWS; ++row) {
        // -- pass 1: the TOP_K-th largest probability, as a bit pattern -------
        // Insertion into a tiny sorted buffer, guarded by a compare against its
        // current tail, so the common case is one unsigned compare per expert.
        uint32_t best[TOP_K];
        for (uint32_t i = 0; i < TOP_K; ++i) {
            best[i] = 0;
        }
        for (uint32_t e = 0; e < E_TOTAL; ++e) {
            const uint32_t t = e >> 5, c = e & 31u;
            const uint32_t off = t * (PROBS_TILE_BYTES / (PROBS_BF16 ? 2 : 4))
                                 + tile_off(row, c);
            const uint32_t bits = PROBS_BF16 ? bf16_to_f32_bits(p16[off]) : p32[off];
            if (bits <= best[TOP_K - 1]) {
                continue;
            }
            uint32_t j = TOP_K - 1;
            while (j > 0 && best[j - 1] < bits) {
                best[j] = best[j - 1];
                --j;
            }
            best[j] = bits;
        }
        const uint32_t threshold = best[TOP_K - 1];

        // -- pass 2: the normalising sum over everything at or above it -------
        // `>=` and not `>`: the chain this replaces thresholds inclusively and
        // therefore admits ties, and the model's output depends on that.
        float total = 0.0f;
        for (uint32_t e = 0; e < E_TOTAL; ++e) {
            const uint32_t t = e >> 5, c = e & 31u;
            const uint32_t off = t * (PROBS_TILE_BYTES / (PROBS_BF16 ? 2 : 4))
                                 + tile_off(row, c);
            const uint32_t bits = PROBS_BF16 ? bf16_to_f32_bits(p16[off]) : p32[off];
            if (bits >= threshold) {
                total += bits_to_f32(bits);
            }
        }
        const float inv = total > 0.0f ? 1.0f / total : 0.0f;

        // -- pass 3: this device's K_SEL largest kept experts ------------------
        uint32_t sel_bits[K_SEL];
        uint16_t sel_id[K_SEL];
        for (uint32_t i = 0; i < K_SEL; ++i) {
            sel_bits[i] = 0;
            sel_id[i] = 0;
        }
        for (uint32_t e = e_lo; e < e_hi; ++e) {
            const uint32_t t = e >> 5, c = e & 31u;
            const uint32_t off = t * (PROBS_TILE_BYTES / (PROBS_BF16 ? 2 : 4))
                                 + tile_off(row, c);
            const uint32_t bits = PROBS_BF16 ? bf16_to_f32_bits(p16[off]) : p32[off];
            if (bits < threshold || bits <= sel_bits[K_SEL - 1]) {
                continue;
            }
            uint32_t j = K_SEL - 1;
            while (j > 0 && sel_bits[j - 1] < bits) {
                sel_bits[j] = sel_bits[j - 1];
                sel_id[j] = sel_id[j - 1];
                --j;
            }
            sel_bits[j] = bits;
            sel_id[j] = (uint16_t)(e - e_lo);
        }

        for (uint32_t i = 0; i < K_SEL; ++i) {
            const float w = sel_bits[i] == 0 ? 0.0f : bits_to_f32(sel_bits[i]) * inv;
            const uint32_t wb = f32_to_bits(w);
            const uint32_t off = tile_off(row, i);
            if (VALS_BF16) {
                v16[off] = f32_bits_to_bf16(wb);
            } else {
                v32[off] = wb;
            }
            // The gather reads the index tile's first face as a flat uint16
            // run, so slot i sits at element i of row 0 -- which is only the
            // same thing as tile_off(row, i) while row is 0. Rows past the
            // first are written where a tile would keep them, and the gather
            // (which is a decode-only path, M=1) reads row 0.
            o16[tile_off(row, i)] = sel_id[i];
        }
    }

    noc_async_write_page(0, v_acc, vals_l1);
    noc_async_write_page(0, i_acc, idx_l1);
    noc_async_write_barrier();
}
