// gnorm1_writer.cpp -- the normalised stream, and this device's own group.
//
// The coordination is in the reader; this only drains what phase 3 packs.
// `mesh_partition(normed, dim=-1)` hands device d columns [d*H, (d+1)*H), which
// is group d -- exactly the tiles a group's cores are already writing, so they
// write them a second time and the partition does not happen.
//
// Compile-time args: 0 NT_G, 1 LOCAL, 2.. TensorAccessorArgs for out, local, devid
// Runtime args: 0 out, 1 local, 2 devid, 3 tile_lo, 4 tile_len, 5 group

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t NT_G = get_compile_time_arg_val(0);
    constexpr uint32_t LOCAL = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = 6, cb_dev = 7;

    const uint32_t o_addr = get_arg_val<uint32_t>(0);
    const uint32_t l_addr = get_arg_val<uint32_t>(1);
    const uint32_t d_addr = get_arg_val<uint32_t>(2);
    const uint32_t lo = get_arg_val<uint32_t>(3);
    const uint32_t len = get_arg_val<uint32_t>(4);
    const uint32_t group = get_arg_val<uint32_t>(5);

    constexpr auto o_ta = TensorAccessorArgs<2>();
    const auto o_acc = TensorAccessor(o_ta, o_addr);
    constexpr auto l_ta = TensorAccessorArgs<o_ta.next_compile_time_args_offset()>();
    const auto l_acc = TensorAccessor(l_ta, l_addr);
    constexpr auto d_ta = TensorAccessorArgs<l_ta.next_compile_time_args_offset()>();
    const auto d_acc = TensorAccessor(d_ta, d_addr);

    if (len == 0) {
        return;
    }
    uint32_t dev = 0;
    if constexpr (LOCAL) {
        const uint32_t dl1 = (get_write_ptr(cb_dev) + 63u) & ~63u;
        noc_async_read_page(0, d_acc, dl1);
        noc_async_read_barrier();
        dev = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dl1);
    }

    for (uint32_t j = 0; j < len; ++j) {
        cb_wait_front(cb_out, 1);
        const uint32_t p = get_read_ptr(cb_out);
        const uint32_t c = lo + j;
        noc_async_write_page(c, o_acc, p);
        if constexpr (LOCAL) {
            if (group == dev) {
                noc_async_write_page(c - dev * NT_G, l_acc, p);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
