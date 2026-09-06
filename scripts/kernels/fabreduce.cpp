// fabreduce.cpp -- the **writer** of a two-chip fabric reduce.
//
// One kernel per processor: this is the writer (BRISC). The sender does its
// fabric send here; the accumulator writes its summed tiles out here. The
// semaphore wait lives in the reader, because the reader is what must not
// touch `scratch` before the payload lands.
//
// Handoff 46.1 item 3. 45.40 settled the shape: hop distance is free, so the
// 4-chip reduce is a chain. What is *not* yet proven is the part the probe never
// did -- receiving a partial and **adding** it to your own. This does exactly
// that on two chips and checks the sum, because a reduce that is fast and wrong
// is worth nothing (45.27 is this session's lesson on that).
//
// chip 0 (ROLE_SEND): ship `NTILES` tiles of `src` to chip 1's `scratch`, and
//                     fused-atomic-inc its semaphore in the same packet.
// chip 1 (ROLE_ACC):  wait on the semaphore, then `out = src + scratch`.
// chips 2, 3:         idle.
//
// Two kernels a chip: this dataflow one, and `fabreduce_compute.cpp` on the
// accumulator. The adder is SFPU rather than a matmul -- no contraction, so
// none of 45.32's padding hazard applies.
//
// Compile-time args: 0 ROLE (0 idle, 1 send, 2 accumulate), 1 BYTES, 2 SEM_ID,
//                    3 NTILES, 4 HOPS, 5.. TensorAccessorArgs for `out`
// Runtime args, sender: 0 src_l1, 1 dst_noc_x, 2 dst_noc_y, 3 dst_l1,
//                       4 sem_noc_x, 5 sem_noc_y, 6.. fabric connection args
// Runtime args, accumulator: 0 out_addr

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"

using namespace tt::tt_fabric::linear::experimental;

void kernel_main() {
    constexpr uint32_t ROLE = get_compile_time_arg_val(0);
    constexpr uint32_t BYTES = get_compile_time_arg_val(1);
    constexpr uint32_t SEM_ID = get_compile_time_arg_val(2);
    constexpr uint32_t HOPS = get_compile_time_arg_val(4);

    if constexpr (ROLE == 0) {
        return;
    } else if constexpr (ROLE == 2) {
        // Accumulator: drain the summed tiles the compute kernel produced.
        constexpr uint32_t NT = get_compile_time_arg_val(3);
        constexpr uint32_t cb_out = 16;
        const uint32_t out_addr = get_arg_val<uint32_t>(0);
        const uint32_t page = BYTES / NT;
        constexpr auto o_ta = TensorAccessorArgs<5>();
        const auto o_acc = TensorAccessor(o_ta, out_addr);
        for (uint32_t i = 0; i < NT; ++i) {
            cb_wait_front(cb_out, 1);
            noc_async_write_page(i, o_acc, get_read_ptr(cb_out));
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
        return;
    } else {
        const uint32_t src_l1 = get_arg_val<uint32_t>(0);
        const uint32_t dst_x = get_arg_val<uint32_t>(1);
        const uint32_t dst_y = get_arg_val<uint32_t>(2);
        const uint32_t dst_l1 = get_arg_val<uint32_t>(3);
        const uint32_t sem_x = get_arg_val<uint32_t>(4);
        const uint32_t sem_y = get_arg_val<uint32_t>(5);

        size_t arg_idx = 6;
        auto sender =
            tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);

        volatile PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
        const uint64_t dst = get_noc_addr(dst_x, dst_y, dst_l1);
        const uint64_t sem = get_noc_addr(sem_x, sem_y, (uint32_t)get_semaphore(SEM_ID));

        sender.open();
        fabric_unicast_noc_fused_unicast_with_atomic_inc(
            &sender, hdr, src_l1, BYTES,
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, sem, 1, true},
            /*num_hops=*/HOPS);
        noc_async_writes_flushed();
        sender.close();
    }
}
