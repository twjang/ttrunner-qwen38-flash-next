// fabprobe.cpp -- what does one fabric hop cost?
//
// Handoff 45.19 prices the 84 wide collectives at ~2.2 ms in model, ~26 us
// apiece, and the only way past that is a single-launch reduce of our own. That
// is a large build with cross-device semaphores, so this measures the one number
// that decides whether it is worth starting: **the cost of a single chip-to-chip
// packet from inside a `generic_op`.**
//
// Chip 0 sends `PAYLOAD` bytes to chip 1 and, in the same packet, atomically
// increments a semaphore there. Chip 1 waits for it. Chips 2 and 3 do nothing.
// Time the op against an otherwise identical launch with ROLE_IDLE everywhere,
// and the difference is one hop plus the fabric open/close.
//
// Compile-time args: 0 ROLE (0 idle, 1 send, 2 recv), 1 PAYLOAD, 2 SEM_ID,
//                    3 HOPS -- how far along the line the packet travels.
// Sweeping HOPS says whether a line reduce should be a chain (linear cost)
// or a tree (fixed cost dominates). Handoff 46.1 item 3.
// Runtime args, sender:   0 src_l1, 1 dst_noc_x, 2 dst_noc_y, 3 dst_l1,
//                         4 sem_noc_x, 5 sem_noc_y, 6.. fabric connection args
// Runtime args, receiver: 0 expected

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"

// The linear-fabric send functions live here, as `minimal_default_writer.cpp`
// also does it.
using namespace tt::tt_fabric::linear::experimental;

void kernel_main() {
    constexpr uint32_t ROLE = get_compile_time_arg_val(0);
    constexpr uint32_t PAYLOAD = get_compile_time_arg_val(1);
    constexpr uint32_t SEM_ID = get_compile_time_arg_val(2);
    constexpr uint32_t HOPS = get_compile_time_arg_val(3);

    if constexpr (ROLE == 0) {
        return;                                   // idle chips: the control arm
    } else if constexpr (ROLE == 2) {
        const uint32_t expected = get_arg_val<uint32_t>(0);
        volatile tt_l1_ptr uint32_t* sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_ID));
        noc_semaphore_wait(sem, expected);
        noc_semaphore_set(sem, 0);
        return;
    } else {
        const uint32_t src_l1 = get_arg_val<uint32_t>(0);
        const uint32_t dst_x = get_arg_val<uint32_t>(1);
        const uint32_t dst_y = get_arg_val<uint32_t>(2);
        const uint32_t dst_l1 = get_arg_val<uint32_t>(3);
        const uint32_t sem_x = get_arg_val<uint32_t>(4);
        const uint32_t sem_y = get_arg_val<uint32_t>(5);

        // `ttnn.setup_fabric_connection` returns exactly the argument block
        // `WorkerToFabricEdmSender::build_from_args` consumes, so it is spliced
        // on at index 6 and read from there.
        size_t arg_idx = 6;
        auto sender =
            tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);

        volatile PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
        const uint64_t dst = get_noc_addr(dst_x, dst_y, dst_l1);
        const uint64_t sem = get_noc_addr(sem_x, sem_y, (uint32_t)get_semaphore(SEM_ID));

        sender.open();
        // Payload and the arrival signal in one packet: a separate atomic inc
        // would be a second hop and would price the wrong thing.
        fabric_unicast_noc_fused_unicast_with_atomic_inc(
            &sender, hdr, src_l1, PAYLOAD,
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, sem, 1, true},
            /*num_hops=*/HOPS);
        noc_async_writes_flushed();
        sender.close();
    }
}
