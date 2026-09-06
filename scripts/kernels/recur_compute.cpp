// recur_compute.cpp -- the whole delta-rule recurrence for one (head, column).
//
//     decayed[i]   = state[h,i,j] * g_exp[h]
//     predicted    = sum_i  k[h,i] @ decayed[i]
//     q_decayed    = sum_i  q[h,i] @ decayed[i]
//     qk           = sum_i  reduce_row(q[h,i] * k[h,i])
//     delta        = (v[h,j] - predicted) * beta[h]
//     out[h,j]     = q_decayed + qk * delta
//     state[h,i,j] = decayed[i] + kt[h,i] @ delta
//
// `decayed` never leaves L1: today it is a 786 KB tensor written by one launch
// and read by three more, and that materialisation -- not the launch count -- is
// where `decode_step`'s 2.95 ms sits (handoff 45.23; 82 us a layer against 4 us
// of state bytes, and invariant 101 prices a marginal launch at 0.8 us).
//
// **One broadcast type only.** g_exp, beta and qk all go through
// `mul_tiles_bcast_scalar`; the outer product `kt (x) delta` is written as a
// `matmul_tiles` of a [32,1] column against a [1,32] row rather than the
// cols-and-rows broadcast pair it looks like, because invariant 108 says two
// broadcast types in one compute kernel hang whatever the init. The zeros in
// both tiles make the matmul exactly the outer product.
//
// FPU matmul and SFPU arithmetic never share a `tile_regs_acquire` (invariant 87).
//
// Compile-time args: 0 DKT, 1 STAGE
// Runtime args: 0 active
//
// STAGE bisects the kernel, because it compiled clean and hung on its first run
// and invariant 109 says bisect the kernel rather than the surroundings. Every
// stage still fills cb_out and cb_snew so the writer completes and the run ends:
//
//   0  no arithmetic at all: state -> snew, v -> out. Reader, CBs, page
//      indices and the writer, and nothing else. If this hangs the fault is
//      plumbing; if it runs, stage 1's broadcast is the first suspect.
//   1  the decayed multiply only            (bcast scalar)
//   2  + the two state matmuls              (adds FPU matmul)
//   3  + qk                                 (adds SFPU reduce)
//   4  + delta                              (adds SFPU sub, second bcast use)
//   5  + out                                (adds SFPU add)
//   6  + the state write-back               (adds matmul again, after SFPU)
//
// Nobody in this project has mixed matmul, broadcast and SFPU in one compute
// kernel before; 2 and 6 are where that first happens.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_hw_startup.h"

// Copy `n` tiles straight through, so an early stage still writes a state
// column and the writer's `cb_wait_front` is satisfied.
inline void copy_to(uint32_t src, uint32_t dst, uint32_t n) {
    init_sfpu(src, dst);
    cb_reserve_back(dst, n);
    for (uint32_t i = 0; i < n; ++i) {
        tile_regs_acquire();
        copy_tile(src, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dst, i);
        tile_regs_release();
    }
    cb_push_back(dst, n);
}

// One tile to `out`, plus a state column, then stop.
inline void finish(uint32_t osrc, uint32_t ssrc, uint32_t ocb, uint32_t scb,
                   uint32_t n) {
    cb_wait_front(osrc, 1);
    init_sfpu(osrc, ocb);
    cb_reserve_back(ocb, 1);
    tile_regs_acquire();
    copy_tile(osrc, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, ocb);
    tile_regs_release();
    cb_push_back(ocb, 1);
    copy_to(ssrc, scb, n);
}

void kernel_main() {
    constexpr uint32_t DKT = get_compile_time_arg_val(0);
    constexpr uint32_t STAGE = get_compile_time_arg_val(1);
    constexpr uint32_t cb_s = 0, cb_q = 1, cb_k = 2, cb_v = 3, cb_g = 4, cb_b = 5,
                       cb_kt = 6, cb_mask = 15, cb_ktm = 18;
    constexpr uint32_t cb_dec = 7, cb_pred = 8, cb_qdec = 9, cb_qk = 10,
                       cb_diff = 11, cb_delta = 12, cb_upd = 13, cb_op = 14;
    constexpr uint32_t cb_out = 16, cb_snew = 17;

    const uint32_t active = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_s, cb_g, cb_dec);
    if (active == 0) {
        return;
    }

    cb_wait_front(cb_s, DKT);
    if constexpr (STAGE < 1) {
        finish(cb_v, cb_s, cb_out, cb_snew, DKT);
        return;
    }

    // ---- decayed[i] = state[i] * g_exp ------------------------------------
    cb_wait_front(cb_g, 1);
    init_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::SCALAR>(cb_s, cb_g, cb_dec);
    cb_reserve_back(cb_dec, DKT);
    for (uint32_t i = 0; i < DKT; ++i) {
        tile_regs_acquire();
        mul_tiles_bcast_scalar(cb_s, cb_g, i, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_dec, i);
        tile_regs_release();
    }
    cb_push_back(cb_dec, DKT);
    cb_wait_front(cb_dec, DKT);

    // Every early exit hands the writer a full set of pages so the run ends and
    // a hang is a hang rather than a deadlock in the harness.
    if constexpr (STAGE < 2) {
        finish(cb_v, cb_dec, cb_out, cb_snew, DKT);
        return;
    }

    // ---- predicted and q_decayed: two reductions over the same column ------
    // Two-arg init and this exact order, because that is what `ksgemv_compute`
    // does and it is the only matmul accumulation in this project known to be
    // right. The three-arg form compiled but is not the tested one.
    cb_wait_front(cb_k, DKT);
    matmul_init(cb_k, cb_dec);
    tile_regs_acquire();
    for (uint32_t i = 0; i < DKT; ++i) {
        matmul_tiles(cb_k, cb_dec, i, i, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_pred, 1);
    pack_tile(0, cb_pred);
    cb_push_back(cb_pred, 1);
    tile_regs_release();

    cb_wait_front(cb_q, DKT);
    matmul_init(cb_q, cb_dec);
    tile_regs_acquire();
    for (uint32_t i = 0; i < DKT; ++i) {
        matmul_tiles(cb_q, cb_dec, i, i, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_qdec, 1);
    pack_tile(0, cb_qdec);
    cb_push_back(cb_qdec, 1);
    tile_regs_release();

    if constexpr (STAGE < 3) {
        finish(cb_pred, cb_dec, cb_out, cb_snew, DKT);
        return;
    }

    // ---- qk = sum_d q[d] * k[d], row 0's total left at element (0, 0) ------
    init_sfpu(cb_q, cb_qk);
    tile_regs_acquire();
    for (uint32_t i = 0; i < DKT; ++i) {
        copy_tile(cb_q, i, 0);
        copy_tile(cb_k, i, 1);
        mul_binary_tile_init();
        if (i == 0) {
            mul_binary_tile(0, 1, 2);
        } else {
            mul_binary_tile(0, 1, 3);
            add_binary_tile_init();
            add_binary_tile(2, 3, 2);
        }
    }
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(2, 1, 1);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_qk, 1);
    pack_tile(2, cb_qk);
    cb_push_back(cb_qk, 1);
    tile_regs_release();

    if constexpr (STAGE < 4) {
        finish(cb_qk, cb_dec, cb_out, cb_snew, DKT);
        return;
    }

    // ---- delta = (v - predicted) * beta ------------------------------------
    cb_wait_front(cb_v, 1);
    cb_wait_front(cb_pred, 1);
    init_sfpu(cb_v, cb_diff);
    tile_regs_acquire();
    copy_tile(cb_v, 0, 0);
    copy_tile(cb_pred, 0, 1);
    sub_binary_tile_init();
    sub_binary_tile(0, 1, 2);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_diff, 1);
    pack_tile(2, cb_diff);
    cb_push_back(cb_diff, 1);
    tile_regs_release();

    cb_wait_front(cb_diff, 1);
    cb_wait_front(cb_b, 1);
    mul_tiles_bcast_scalar_init_short(cb_diff, cb_b);
    tile_regs_acquire();
    mul_tiles_bcast_scalar(cb_diff, cb_b, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_delta, 1);
    pack_tile(0, cb_delta);
    cb_push_back(cb_delta, 1);
    tile_regs_release();
    cb_wait_front(cb_delta, 1);
    cb_wait_front(cb_qk, 1);

    if constexpr (STAGE < 5) {
        finish(cb_delta, cb_dec, cb_out, cb_snew, DKT);
        return;
    }

    // ---- out = q_decayed + qk * delta --------------------------------------
    mul_tiles_bcast_scalar_init_short(cb_delta, cb_qk);
    tile_regs_acquire();
    mul_tiles_bcast_scalar(cb_delta, cb_qk, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_upd, 1);
    pack_tile(0, cb_upd);
    cb_push_back(cb_upd, 1);
    tile_regs_release();

    cb_wait_front(cb_upd, 1);
    cb_wait_front(cb_qdec, 1);
    init_sfpu(cb_qdec, cb_out);
    tile_regs_acquire();
    copy_tile(cb_qdec, 0, 0);
    copy_tile(cb_upd, 0, 1);
    add_binary_tile_init();
    add_binary_tile(0, 1, 2);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(2, cb_out);
    cb_push_back(cb_out, 1);
    tile_regs_release();
    cb_pop_front(cb_upd, 1);

    if constexpr (STAGE < 6) {
        copy_to(cb_dec, cb_snew, DKT);
        return;
    }

    // ---- state[i] = decayed[i] + kt[i] @ delta ------------------------------
    // The matmul is the outer product: kt[i] is a column, delta a row, and the
    // zeros in both make `sum_c kt[a,c] * delta[c,b]` collapse to
    // kt[a,0] * delta[0,b]. Matmul and the SFPU add take separate windows.
    cb_wait_front(cb_kt, DKT);

    // All four outer products, then all four adds -- **not** alternating. The
    // first version ran matmul_init and init_sfpu once per i, four transitions
    // between the FPU matmul and the SFPU in one kernel, and it hung where
    // STAGE 5 (which makes the transition once, like `ksgemv_compute`) runs
    // clean. One window each is the shape this project has working.
    // Its **own** buffer, not `cb_upd`. cb_upd has already been pushed and
    // popped once for qk*delta, so its pointers have advanced; reserving DKT
    // more and then addressing them with an indexed `pack_tile` is an indexed
    // write into a wrapped circular buffer, and that is where the restructured
    // write-back hung two runs out of two against a 12 % base rate.
    // **Mask kt's pad columns first** (handoff 45.32, invariant 150). The outer
    // product below is a matmul, and `matmul_tiles` contracts all 32 columns --
    // `sum_c kt[a,c] * delta[c,b]` only collapses to `kt[a,0] * delta[0,b]` if
    // one operand's padding is zero. `kt` reaches this kernel through
    // `ttnn.transpose` on a recycled buffer, so its pad columns hold whatever
    // was there before; in the model that made the *state* infinite while the
    // output, which never contracts over the pad, stayed correct to 9.6e-04.
    // `cb_mask` is 1 in column 0 and 0 elsewhere, so `cb_ktm` has exactly the
    // column the outer product needs and nothing else.
    cb_wait_front(cb_mask, 1);
    init_sfpu(cb_kt, cb_ktm);
    cb_reserve_back(cb_ktm, DKT);
    for (uint32_t i = 0; i < DKT; ++i) {
        tile_regs_acquire();
        copy_tile(cb_kt, i, 0);
        copy_tile(cb_mask, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 2);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(2, cb_ktm, i);
        tile_regs_release();
    }
    cb_push_back(cb_ktm, DKT);
    cb_wait_front(cb_ktm, DKT);

    // STAGE 7 is a probe, not a stage of the computation: it writes the masked
    // `kt` tiles out as the new state, so a `to_torch` of the state shows all 32
    // columns of what the kernel actually reads. The state is [Dk, Dv] and fully
    // real, so nothing is dropped the way `to_torch` drops a [Dk, 1]'s padding.
    if constexpr (STAGE == 7) {
        finish(cb_v, cb_ktm, cb_out, cb_snew, DKT);
        return;
    }

    matmul_init(cb_ktm, cb_delta);
    cb_reserve_back(cb_op, DKT);
    for (uint32_t i = 0; i < DKT; ++i) {
        tile_regs_acquire();
        matmul_tiles(cb_ktm, cb_delta, i, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_op, i);
        tile_regs_release();
    }
    cb_push_back(cb_op, DKT);
    cb_wait_front(cb_op, DKT);

    init_sfpu(cb_dec, cb_snew);
    cb_reserve_back(cb_snew, DKT);
    for (uint32_t i = 0; i < DKT; ++i) {
        tile_regs_acquire();
        copy_tile(cb_dec, i, 0);
        copy_tile(cb_op, i, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 2);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(2, cb_snew, i);
        tile_regs_release();
    }
    cb_push_back(cb_snew, DKT);
}
