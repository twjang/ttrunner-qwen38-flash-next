// nop_reader.cpp -- the cheapest thing a generic_op can do.
//
// Exists to price the *launch* rather than the work: if a program that touches
// nothing still costs what a ttnn elementwise op costs, then the 5.8 us floor is
// dispatch and no kernel can get under it. If it is much cheaper, the gap is
// what ttnn's ops spend on top of dispatch, and 5764 of those a step is worth
// knowing about.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // read one page so the program is not optimised into nothing
    const uint32_t addr = get_arg_val<uint32_t>(0);
    constexpr auto ta = TensorAccessorArgs<0>();
    const auto acc = TensorAccessor(ta, addr);
    noc_async_read_page(0, acc, get_write_ptr(0));
    noc_async_read_barrier();
}
