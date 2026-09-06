"""What does one fabric hop cost from inside a `generic_op`?

    uv run python scripts/dev/fabric_probe.py [payload_bytes]

Handoff 45.19 prices the 84 wide collectives at ~2.2 ms in model -- ~26 us each
-- and 45.20 shows part of that is real wire time rather than dispatch. The only
way past it is a single-launch reduce of our own, which is a large build with
cross-device semaphores on hardware where a mistake wedges the cards and
`TT_METAL_WATCHER` aborts (invariant 103). So this measures the one number that
decides whether to start: **one chip-to-chip packet, sent from a generic_op.**

Chip 0 sends the payload to chip 1 and fused-atomic-incs a semaphore there; chip
1 waits on it; chips 2 and 3 run the same kernel with ROLE_IDLE and return
immediately. The control arm is every chip idle, so the difference is one hop
plus the fabric open/close.

If a hop is not comfortably under the ~26 us a reduce costs today, the kernel is
not worth building and this cost twenty minutes instead of a day.
"""
import sys
import time
from pathlib import Path

import torch
import ttnn
import ttnn._ttnn.fabric as fab

REPO = Path("/home/twjang/twtest")
KDIR = REPO / "scripts" / "kernels"
PAYLOAD = int(sys.argv[1]) if len(sys.argv) > 1 else 5120     # a 2560-wide bf16 row
REPS = 200
WORKER = ttnn.CoreCoord(0, 0)


def build(mesh, role_by_chip, src, dst):
    """A MeshProgramDescriptor: one ProgramDescriptor per chip."""
    # This build's MeshDevice has no `get_devices`; ops.py carries the same
    # hasattr fallback, so the mesh itself answers the coordinate query.
    d0 = mesh.get_devices()[0] if hasattr(mesh, "get_devices") else mesh
    phys = d0.worker_core_from_logical_core(WORKER)
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(WORKER, WORKER)])
    mesh_id = fab.get_all_fabric_mesh_ids()[0]
    defines = dict(ttnn.get_fabric_kernel_defines("Linear"))

    programs = {}
    for chip, role in enumerate(role_by_chip):
        # Semaphore 0 on every chip: the receiver waits on it, the sender's
        # packet increments it remotely.
        pd = ttnn.ProgramDescriptor(
            kernels=[], semaphores=[
                ttnn.SemaphoreDescriptor(id=0, core_ranges=crs, initial_value=0)],
            cbs=[])
        rt = [0]
        if role == 1:
            # setup_fabric_connection MUTATES pd (appends its semaphores) and
            # returns the argument block the kernel's build_from_args consumes.
            fargs = ttnn.setup_fabric_connection(
                fab.FabricNodeId(mesh_id, chip), fab.FabricNodeId(mesh_id, chip + 1),
                0, pd, WORKER, ttnn.CoreType.WORKER)
            rt = [src.buffer_address(), phys.x, phys.y, dst.buffer_address(),
                  phys.x, phys.y] + list(fargs)
        elif role == 2:
            rt = [1]
        kd = ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "fabprobe.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs,
            compile_time_args=[role, PAYLOAD, 0],
            defines=list(defines.items()),
            runtime_args=[(WORKER, rt)],
            config=ttnn.WriterConfigDescriptor())
        pd.kernels = [kd]
        # Keyed by a *range*, not a coordinate: the descriptor is a list of
        # (MeshCoordinateRange, ProgramDescriptor) pairs, and a bare coordinate
        # comes back as std::bad_cast.
        at = ttnn.MeshCoordinate(0, chip)
        programs[ttnn.MeshCoordinateRange(at, at)] = pd
    return ttnn.MeshProgramDescriptor(programs)


def time_it(mesh, io, mpd, label):
    for _ in range(5):
        ttnn.generic_op(io, mpd)
    ttnn.synchronize_device(mesh)
    best = []
    for _ in range(9):
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(REPS):
            ttnn.generic_op(io, mpd)
        ttnn.synchronize_device(mesh)
        best.append(1e6 * (time.perf_counter() - t0) / REPS)
    best.sort()
    us = best[len(best) // 2]
    print(f"RESULT {label:22s} {us:8.2f} us", flush=True)
    return us


def main() -> None:
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    try:
        rep = ttnn.ReplicateTensorToMesh(mesh)
        src = ttnn.from_torch(torch.randn(1, 1, 32, 2560), dtype=ttnn.bfloat16,
                              layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        dst = ttnn.from_torch(torch.zeros(1, 1, 32, 2560), dtype=ttnn.bfloat16,
                              layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        io = [src, dst]
        # An eager `generic_op` launch is ~84 us of dispatch, which buries a hop
        # entirely -- the first run of this measured the hop at *minus* 1.67 us.
        # So compare against `ttnn.all_reduce` measured **the same way**, in the
        # same process: both then carry the same dispatch and it cancels.
        idle = time_it(mesh, io, build(mesh, [0, 0, 0, 0], src, dst), "launch only (control)")
        hop = time_it(mesh, io, build(mesh, [1, 2, 0, 0], src, dst), f"one hop, {PAYLOAD} B")

        row = ttnn.from_torch(torch.randn(1, 1, 32, 2560), dtype=ttnn.bfloat16,
                              layout=ttnn.TILE_LAYOUT, device=mesh,
                              mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        for _ in range(5):
            ttnn.all_reduce(row, cluster_axis=1, topology=ttnn.Topology.Linear, num_links=3)
        ttnn.synchronize_device(mesh)
        best = []
        for _ in range(9):
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for _ in range(REPS):
                ttnn.all_reduce(row, cluster_axis=1, topology=ttnn.Topology.Linear, num_links=3)
            ttnn.synchronize_device(mesh)
            best.append(1e6 * (time.perf_counter() - t0) / REPS)
        best.sort()
        ar = best[len(best) // 2]
        print(f"RESULT {'ttnn.all_reduce 2560w':22s} {ar:8.2f} us", flush=True)

        print(f"RESULT hop - control = {hop - idle:+.2f} us  "
              f"(a launch is {idle:.0f} us, so this is noise, not the hop)", flush=True)
        print(f"RESULT what matters: one fabric packet from a generic_op costs "
              f"{hop:.1f} us end to end against {ar:.1f} for the op it would "
              f"replace -- ratio {ar / hop:.2f}x", flush=True)
        print("RESULT a 4-chip reduce is more than one packet, so treat this as "
              "the optimistic bound; if it is not well above 1x, stop here",
              flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
