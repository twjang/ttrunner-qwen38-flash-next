// group_scale_writer.cpp -- the normalised stream, and this device's own group.
//
// `mesh_partition(normed, dim=-1)` hands device d columns [d*H, (d+1)*H), which
// is group d -- exactly the tiles this kernel is already writing. So it writes
// them a second time into `local` and the partition, 5.38 us a call, does not
// happen.
//
// `generic_op` broadcasts one program to the whole mesh, so runtime args cannot
// say which device is running. A tensor sharded on dim 0 can, which is how
// `_router_devid` does it.
//
// Compile-time args: 0 NT_G, 1 LOCAL (1: also write this device's group),
//                    2.. TensorAccessorArgs for out, local, devid
// Runtime args: 0 out_addr, 1 local_addr, 2 devid_addr, 3 lo, 4 hi

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t NT_G = get_compile_time_arg_val(0);
    constexpr uint32_t LOCAL = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = 4, cb_dev = 5;

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t loc_addr = get_arg_val<uint32_t>(1);
    const uint32_t dev_addr = get_arg_val<uint32_t>(2);
    const uint32_t lo = get_arg_val<uint32_t>(3);
    const uint32_t hi = get_arg_val<uint32_t>(4);

    constexpr auto o_ta = TensorAccessorArgs<2>();
    const auto o_acc = TensorAccessor(o_ta, out_addr);
    constexpr auto l_ta = TensorAccessorArgs<o_ta.next_compile_time_args_offset()>();
    const auto l_acc = TensorAccessor(l_ta, loc_addr);
    constexpr auto d_ta = TensorAccessorArgs<l_ta.next_compile_time_args_offset()>();
    const auto d_acc = TensorAccessor(d_ta, dev_addr);

    uint32_t dev = 0;
    if constexpr (LOCAL) {
        const uint32_t dl1 = (get_write_ptr(cb_dev) + 63u) & ~63u;
        noc_async_read_page(0, d_acc, dl1);
        noc_async_read_barrier();
        dev = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dl1);
    }

    for (uint32_t c = lo; c < hi; ++c) {
        cb_wait_front(cb_out, 1);
        const uint32_t p = get_read_ptr(cb_out);
        noc_async_write_page(c, o_acc, p);
        if constexpr (LOCAL) {
            if (c / NT_G == dev) {
                noc_async_write_page(c - dev * NT_G, l_acc, p);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
