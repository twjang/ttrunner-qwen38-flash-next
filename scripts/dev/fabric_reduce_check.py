"""A two-chip fabric reduce, checked for correctness before anything else.

    uv run python scripts/dev/fabric_reduce_check.py

Handoff 46.1 item 3. 45.40 settled the design -- hop distance is free, so the
4-chip reduce is a chain -- and `fabric_probe.py` proved a `generic_op` can send
a packet. What neither did is the part that makes it a *reduce*: receive a
partial and add it to your own. This does that across two chips and checks the
sum against torch.

Correctness first, deliberately. 45.27 is this session's lesson: a fused kernel
that is fast and silently wrong costs more than it saves, and it was a *state*
that was wrong while the output looked fine. So this asserts the arithmetic on
device before any timing is taken.

chip 0 sends its partial to chip 1's scratch and fused-atomic-incs a semaphore
there; chip 1 waits, adds, writes `out`. Chips 2 and 3 idle.
"""
import sys
from pathlib import Path

import torch
import ttnn
import ttnn._ttnn.fabric as fab

KDIR = Path("/home/twjang/twtest/scripts/kernels")
WORKER = ttnn.CoreCoord(0, 0)
NT = 8                                   # tiles; 8 x 32 x 32 bf16 = 16 KB
WIDTH = NT * 32


# The 1x4 mesh is a **ring**, not a line. Probed by trying
# `setup_fabric_connection` on all twelve ordered pairs:
#
#     D0 -> [1, 2]    D1 -> [0, 3]    D2 -> [0, 3]    D3 -> [1, 2]
#
# so the links are 0-1, 0-2, 1-3, 2-3 -- the cycle 0-1-3-2-0 -- and **D1 -> D2
# does not exist**, which is what killed the first four-chip attempt with
# `Could not find any forwarding direction from (M0, D1) to (M0, D2)`.
# The chain therefore has to walk the ring, not the index order.
RING = [0, 1, 3, 2]
NEXT = {RING[i]: RING[i + 1] for i in range(len(RING) - 1)}


def build(mesh, roles, src, scratch, out):
    d0 = mesh.get_devices()[0] if hasattr(mesh, "get_devices") else mesh
    phys = d0.worker_core_from_logical_core(WORKER)
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(WORKER, WORKER)])
    mesh_id = fab.get_all_fabric_mesh_ids()[0]
    defines = dict(ttnn.get_fabric_kernel_defines("Linear"))
    page = ttnn.TensorAccessorArgs(src).get_compile_time_args()[1]
    a_src = list(ttnn.TensorAccessorArgs(src).get_compile_time_args())
    a_scr = list(ttnn.TensorAccessorArgs(scratch).get_compile_time_args())
    a_out = list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    nbytes = NT * page

    programs = {}
    for chip, role in enumerate(roles):
        pd = ttnn.ProgramDescriptor(
            kernels=[], semaphores=[
                ttnn.SemaphoreDescriptor(id=0, core_ranges=crs, initial_value=0)],
            cbs=[
                # NT tiles, not 2: the sender ships the whole block in one
                # packet from `cb_mine`, and the packet lands in `cb_theirs`.
                ttnn.CBDescriptor(total_size=NT * page, core_ranges=crs,
                                  format_descriptors=[ttnn.CBFormatDescriptor(
                                      buffer_index=i, data_format=src.dtype,
                                      page_size=page)])
                for i in (0, 1, 16)])
        w_rt, r_rt = [0], [0, 0]
        if role in (1, 3):
            # 1 sends its own partial; 3 receives, adds, and forwards the running
            # sum. Both need the fabric connection, and 3 also needs the
            # semaphore count its reader waits on.
            r_rt = [src.buffer_address(), 1 if role == 3 else 0]
            # Mutates pd; returns the block build_from_args consumes. Connect to
            # the ADJACENT node -- distance is num_hops (invariant 159).
            fargs = ttnn.setup_fabric_connection(
                fab.FabricNodeId(mesh_id, chip), fab.FabricNodeId(mesh_id, NEXT[chip]),
                0, pd, WORKER, ttnn.CoreType.WORKER)
            # src/dst L1 addresses are taken inside the kernel (invariant 161);
            # these slots carry the NOC coordinates and the fabric block.
            w_rt = [0, phys.x, phys.y, 0, phys.x, phys.y] + list(fargs)
        elif role == 2:
            w_rt = [out.buffer_address()]
            r_rt = [src.buffer_address(), 1]

        def kd(name, ct, rt, cfg):
            return ttnn.KernelDescriptor(
                kernel_source=str(KDIR / name),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=crs, compile_time_args=ct,
                defines=list(defines.items()),
                runtime_args=[(WORKER, rt)], config=cfg)

        pd.kernels = [
            kd("fabreduce_reader.cpp", [role, NT, 0] + a_src,
               r_rt, ttnn.ReaderConfigDescriptor()),
            kd("fabreduce_compute.cpp", [role, NT], [0],
               ttnn.ComputeConfigDescriptor()),
            kd("fabreduce.cpp", [role, nbytes, 0, NT, 1] + a_out,
               w_rt, ttnn.WriterConfigDescriptor()),
        ]
        at = ttnn.MeshCoordinate(0, chip)
        programs[ttnn.MeshCoordinateRange(at, at)] = pd
    return ttnn.MeshProgramDescriptor(programs)


def main() -> None:
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    try:
        torch.manual_seed(0)
        # Each chip a different partial, so a wrong sum cannot look right.
        parts = torch.randn(4, 1, 32, WIDTH)
        src = ttnn.from_torch(parts, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                              device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
        zeros = torch.zeros(4, 1, 32, WIDTH)
        scratch = ttnn.from_torch(zeros, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                  device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
        out = ttnn.from_torch(zeros, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                              device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
        print(f"RESULT shapes src {list(src.shape)} page-tiles {NT}", flush=True)

        def check(roles, sink, want, label):
            ttnn.generic_op([src, scratch, out],
                            build(mesh, roles, src, scratch, out))
            ttnn.synchronize_device(mesh)
            got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
            have = got[sink].to(torch.float32)
            err = float((have - want).abs().max())
            print(f"RESULT {label}: err {err:.4e}  |want| "
                  f"{float(want.abs().max()):.4e} |have| "
                  f"{float(have.abs().max()):.4e}  "
                  + ("CORRECT" if err < 8e-2 else "WRONG"), flush=True)

        # Kept as a regression: if the chain extension breaks, this must not
        # quietly keep passing as though the mechanism still worked.
        check([1, 2, 0, 0], 1, (parts[0] + parts[1]).to(torch.float32),
              "two chips,  chip1 = p0+p1")
        # The forward half of the 4-chip chain: 0 -> 1 -> 2 -> 3, each middle
        # chip adding its own partial and passing the running sum on.
        # Roles are indexed by chip, and the chain walks RING = 0 -> 1 -> 3 -> 2,
        # so chip 2 is the one that ends up holding the total.
        roles = [0, 0, 0, 0]
        roles[RING[0]] = 1                      # send
        roles[RING[1]] = 3                      # receive, add, forward
        roles[RING[2]] = 3
        roles[RING[3]] = 2                      # receive, add, write out
        check(roles, RING[3], parts.sum(0).to(torch.float32),
              f"four chips via ring {RING}, chip{RING[3]} = sum(p0..p3)")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
