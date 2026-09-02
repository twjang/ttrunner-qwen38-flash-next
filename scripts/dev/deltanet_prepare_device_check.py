"""Does `prepare_device` produce what `prepare` produces?

The eight tensors feed a device op whose intra-chunk error is already known
(0.36 % at position 0 of a 128-token chunk), so the bar here is the float32
floor, not the op's output: if the inputs match to ~1e-5 relative the port is
faithful and anything downstream is the op's own business.

  uv run python scripts/dev/deltanet_prepare_device_check.py [--heads 12]
"""

import argparse
import torch
import ttnn

from twtest.tt.deltanet import CHUNK, prepare, prepare_device


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    scale = b.abs().max().clamp(min=1e-12)
    return ((a - b).abs().max() / scale).item() * 100.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--chunks", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--correlated", action="store_true",
        help="nearly parallel keys and slow decay -- the regime that broke L_inv",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    h, nc = args.heads, args.chunks
    seq = nc * CHUNK
    q = torch.randn(1, seq, h, CHUNK)
    k = torch.randn(1, seq, h, CHUNK)
    v = torch.randn(1, seq, h, CHUNK)
    if args.correlated:
        # Random normals are the easy case: `l_unit`'s off-diagonal entries come
        # out small and mixed in sign, and every candidate inverse looks fine.
        # Real heads produce a dense, positive, near-1 lower triangle -- which
        # is what caught the Neumann series at 620875 % where random inputs put
        # it at 0.02 %. Nearly parallel keys reproduce it.
        k = torch.randn(1, 1, 1, CHUNK) + 0.05 * torch.randn(1, seq, h, CHUNK)
        q = torch.randn(1, 1, 1, CHUNK) + 0.05 * torch.randn(1, seq, h, CHUNK)
    # g is a log decay: negative, and spanning the range the real model uses
    # (iteration 004 -- one head sits at A = -158, which is why the reference
    # forms every decay by subtraction in log space).
    g = -torch.rand(1, seq, h) * 4.0
    beta = torch.sigmoid(torch.randn(1, seq, h))
    if args.correlated:
        g = -torch.rand(1, seq, h) * 1e-3          # barely any decay
        beta = torch.sigmoid(torch.randn(1, seq, h) * 0.3 + 2.0)

    host = prepare(q, k, v, g, beta)
    host.pop("_meta")

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    try:
        def up(t, width):
            # [1, S, H, W] -> [H, NC, C, W]
            t = t.reshape(1, seq, h, width).permute(2, 0, 1, 3).reshape(h, nc, CHUNK, width)
            return ttnn.from_torch(
                t.contiguous().float(), dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT, device=mesh,
            )

        dev = prepare_device(
            up(q, CHUNK), up(k, CHUNK), up(v, CHUNK),
            up(g.unsqueeze(-1), 1), up(beta.unsqueeze(-1), 1), mesh=mesh,
        )
        print(f"heads={h} chunks={nc} seq={seq} "
              f"{'correlated keys' if args.correlated else 'random keys'}")
        worst = 0.0
        for name, want in host.items():
            # inputs are replicated, so every device computed the same thing;
            # concatenating and keeping the first shard reads one of them.
            got = ttnn.to_torch(
                dev[name], mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
            )[:h]
            want = want.reshape(got.shape)
            err = rel(got, want)
            worst = max(worst, err)
            print(f"  {name:<12} {tuple(got.shape)!s:<22} rel {err:8.5f} %")
        print(f"WORST {worst:.5f} %")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
