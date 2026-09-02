"""The residency plan and the layouts derived from it, without a device."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from twtest.tt.blockfloat import MANTISSA_BITS, round_trip
from twtest.tt.convert import SHARD_DIM, device_layout
from twtest.tt.plan import BYTES_PER_ELEMENT, PLAN, Residency, Shard, plan_for

# One representative name per tensor role in the checkpoint.
ROLES = [
    "token_embd.weight",
    "output.weight",
    "output_hc_norm.weight",
    "output_hc_down.weight",
    "output_hc_up.weight",
    "per_layer_token_embd.weight",
    "blk.0.attn_qkv.weight",
    "blk.0.attn_gate.weight",
    "blk.0.ssm_a",
    "blk.0.ssm_dt.bias",
    "blk.0.ssm_norm.weight",
    "blk.0.ssm_conv1d.weight",
    "blk.0.ssm_alpha.weight",
    "blk.0.ssm_beta.weight",
    "blk.0.ssm_out.weight",
    "blk.3.attn_q.weight",
    "blk.3.attn_k.weight",
    "blk.3.attn_v.weight",
    "blk.3.attn_output.weight",
    "blk.3.attn_q_norm.weight",
    "blk.3.attn_k_norm.weight",
    "blk.3.indexer.q_proj.weight",
    "blk.3.indexer.k_norm.weight",
    "blk.0.ffn_gate_exps.weight",
    "blk.0.ffn_up_exps.weight",
    "blk.0.ffn_down_exps.weight",
    "blk.0.ffn_gate_inp.weight",
    "blk.0.ffn_gate_inp_shexp.weight",
    "blk.0.ffn_gate_shexp.weight",
    "blk.0.ffn_down_shexp.weight",
    "blk.0.hc_attn_norm.weight",
    "blk.0.hc_attn_inject.weight",
    "blk.1.ple_key.weight",
    "blk.1.ple_value.weight",
    "blk.1.ple_norm_conv.weight",
    "blk.1.ple_conv1d.weight",
]


@pytest.mark.parametrize("name", ROLES)
def test_every_role_is_planned(name: str) -> None:
    entry = plan_for(name)
    assert entry.dtype in BYTES_PER_ELEMENT or entry.dtype == "source"
    assert entry.shard in SHARD_DIM


def test_unplanned_name_raises() -> None:
    with pytest.raises(KeyError):
        plan_for("blk.0.some_tensor_that_does_not_exist")


def test_gather_tensors_stay_on_host() -> None:
    """The two big gathers must never be made device-resident.

    per_layer_token_embd is 28.8 GB quantised and 205 GB as f32, and only 16
    rows are touched per token.
    """
    for name in ("per_layer_token_embd.weight", "token_embd.weight"):
        entry = plan_for(name)
        assert entry.residency is Residency.HOST
        assert entry.dtype == "source", "must stay in its GGUF quantisation"


def test_sharding_covers_experts_and_the_deltanet() -> None:
    """What is sharded is a batch-dependent decision, and it changed.

    Replicating the dense weights was right while decode was purely
    dispatch-bound: it halved the collectives (96 -> 48). At batch 32 the
    DeltaNet became ~65% of the step, with every device redundantly computing
    all 48 heads over a 3.6 GB replicated recurrent state, so its weights are
    head-sharded too -- 1.69x throughput, and the smaller state is what lets the
    batch grow past 32. Attention (12 layers, 2 KV heads) stays replicated.
    """
    sharded = {n for n in ROLES if plan_for(n).shard is not Shard.REPLICATE}
    assert sharded == {
        # MoE experts: the bulk of the weights
        "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_up_exps.weight",
        "blk.0.ffn_down_exps.weight",
        # LM head, sharded by vocabulary
        "output.weight",
        # Gated DeltaNet, sharded by head
        "blk.0.attn_qkv.weight",
        "blk.0.attn_gate.weight",
        "blk.0.ssm_conv1d.weight",
        "blk.0.ssm_alpha.weight",
        "blk.0.ssm_beta.weight",
        "blk.0.ssm_a",
        "blk.0.ssm_dt.bias",
        "blk.0.ssm_out.weight",
    }
    # the full-attention layers stay replicated
    for name in ("blk.3.attn_q.weight", "blk.3.attn_k.weight", "blk.3.attn_output.weight"):
        assert plan_for(name).shard is Shard.REPLICATE, name


def test_qkv_split_gives_each_device_the_heads_its_v_heads_pair_with() -> None:
    """A flat split of the 10240 q|k|v axis mispairs heads with correct shapes.

    Chunking q and k four ways is not enough either: V heads are tiled over K
    heads, so global v-head j reads global k-head `j % n_k`, and device d's
    twelve v heads need `(12d + i) % 16` -- not the contiguous four a chunk
    gives it. The converter hands each device exactly those twelve, in matching
    order, so the pairing is local and needs no expansion at all. Chunking cost
    the engine 25.5 % next-token accuracy against the reference's 80.9 %.
    """
    from twtest.tt.convert import split_qkv_channels

    hd, n_k, n_v, n_dev = 128, 16, 48, 4
    key_dim, value_dim = n_k * hd, n_v * hd
    t = torch.arange(2 * key_dim + value_dim, dtype=torch.float32).reshape(1, 1, 1, -1)
    pieces = split_qkv_channels(t, -1, n_dev, key_dim, value_dim, hd)

    n_v_local = n_v // n_dev
    per = n_v_local * hd
    assert all(p.shape[-1] == 3 * per for p in pieces), "q, k and v all hold n_v_local heads"

    for d, p in enumerate(pieces):
        flat = p[0, 0, 0]
        q_heads = [int(flat[i * hd]) // hd for i in range(n_v_local)]
        k_heads = [int(flat[per + i * hd] - key_dim) // hd for i in range(n_v_local)]
        v_heads = [int(flat[2 * per + i * hd] - 2 * key_dim) // hd for i in range(n_v_local)]
        assert v_heads == list(range(d * n_v_local, (d + 1) * n_v_local)), "v is chunked"
        assert q_heads == k_heads, "q and k must follow the same heads"
        for i, v in enumerate(v_heads):
            assert q_heads[i] == v % n_k, f"device {d} head {i} pairs wrongly"

    # a flat chunk would put k-channels in device 0's slice
    assert torch.chunk(t, n_dev, dim=-1)[0][0, 0, 0, -1] >= key_dim


def test_router_and_indexer_keep_precision() -> None:
    """A wrong expert or a wrong token set is not a small perturbation."""
    assert plan_for("blk.0.ffn_gate_inp.weight").dtype == "float32"
    assert plan_for("blk.3.indexer.q_proj.weight").dtype == "bfloat16"


@pytest.mark.parametrize(
    "name,gguf_shape,expected",
    [
        # expert stacks: (E, out, in) -> [1, E, K, N]
        ("blk.0.ffn_gate_exps.weight", (8, 640, 2560), (1, 8, 2560, 640)),
        ("blk.0.ffn_down_exps.weight", (8, 2560, 640), (1, 8, 640, 2560)),
        # linear weights: (out, in) -> (in, out)
        ("blk.0.attn_qkv.weight", (10240, 2560), (1, 1, 2560, 10240)),
        ("output.weight", (1024, 2560), (1, 1, 2560, 1024)),
        # 1-D
        ("blk.0.ssm_a", (48,), (1, 1, 1, 48)),
        # depthwise conv filters keep (channels, kernel)
        ("blk.0.ssm_conv1d.weight", (10240, 4), (1, 1, 10240, 4)),
    ],
)
def test_device_layout(name, gguf_shape, expected) -> None:
    out = device_layout(name, torch.zeros(gguf_shape))
    assert tuple(out.shape) == expected


def test_expert_shard_dims_are_tile_aligned() -> None:
    """Sharding a tiled block-float tensor is only lossless on tile boundaries."""
    for name, shape in (
        ("blk.0.ffn_gate_exps.weight", (512, 640, 2560)),
        ("blk.0.ffn_down_exps.weight", (512, 2560, 640)),
    ):
        laid = device_layout(name, torch.zeros(shape))
        dim = SHARD_DIM[plan_for(name).shard]
        assert laid.shape[dim] % (4 * 32) == 0, f"{name} dim {dim} not 4x32-aligned"


class TestBlockFloat:
    def test_quantisation_is_idempotent(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal((256, 64)).astype(np.float32)
        once = round_trip(x, "bfloat8_b")
        twice = round_trip(once, "bfloat8_b")
        assert np.array_equal(once, twice)

    @pytest.mark.parametrize("dtype,bound", [("bfloat8_b", 0.02), ("bfloat4_b", 0.20)])
    def test_error_within_format_bound(self, dtype: str, bound: float) -> None:
        rng = np.random.default_rng(1)
        x = rng.standard_normal((512, 32)).astype(np.float32)
        err = np.abs(round_trip(x, dtype) - x).mean() / np.abs(x).mean()
        assert err < bound

    def test_error_does_not_shrink_with_contraction_length(self) -> None:
        """The property that ruled bfloat4_b out for gate/up.

        Block-float error is multiplicative, so a dot product's relative error
        stays ~2^-mantissa_bits however long K is. Additive noise would fall as
        1/sqrt(K).
        """
        rng = np.random.default_rng(2)
        errs = []
        for k in (256, 4096):
            w = rng.standard_normal((k, 64)).astype(np.float32)
            x = rng.standard_normal((1, k)).astype(np.float32)
            ref = x @ w
            got = x @ round_trip(w, "bfloat4_b")
            errs.append(float(np.abs(got - ref).std() / np.abs(ref).std()))
        assert errs[1] > errs[0] * 0.5, f"error averaged away unexpectedly: {errs}"

    def test_bfloat16_truncates(self) -> None:
        x = np.array([1.0, 1.0009765625, -3.5], dtype=np.float32)
        out = round_trip(x, "bfloat16")
        assert out.dtype == np.float32
        assert out[0] == 1.0 and out[2] == -3.5

    def test_unknown_dtype_raises(self) -> None:
        with pytest.raises(ValueError, match="int3"):
            round_trip(np.zeros(16, dtype=np.float32), "int3")

    def test_mantissa_bits_match_formats(self) -> None:
        assert MANTISSA_BITS == {"bfloat8_b": 7, "bfloat4_b": 3}


def test_shared_expert_gate_is_a_column() -> None:
    """A 1-D tensor that is actually a linear weight.

    `ffn_gate_inp_shexp` maps hidden_size -> 1 (the shared expert's sigmoid
    gate). Laid out as a row it would either fail the matmul or silently
    broadcast, so it must come out as (.., hidden, 1) while every other 1-D
    tensor -- norm gammas, ssm_a, dt_bias -- stays a row.
    """
    gate = device_layout("blk.0.ffn_gate_inp_shexp.weight", torch.zeros(2560))
    assert tuple(gate.shape) == (1, 1, 2560, 1)
    for name, n in (
        ("blk.0.ssm_a", 48),
        ("blk.0.ssm_dt.bias", 48),
        ("blk.0.ssm_norm.weight", 128),
        ("blk.0.hc_attn_norm.weight", 10240),
    ):
        assert tuple(device_layout(name, torch.zeros(n)).shape) == (1, 1, 1, n), name


def test_head_expansion_is_tiled_and_per_sequence() -> None:
    """V heads are stored *tiled* over K heads: v-head j reads k-head j % n_k.

    Upstream's HF code interleaves instead
    (`query.repeat_interleave(num_v_heads // num_k_heads, dim=2)`), and taking
    that at face value cost a day: the GGUF converter permutes the head order,
    so the two are not the same model on this checkpoint. The arbiter is the
    float32 reference's next-token accuracy on real text -- 80.9 % tiled against
    12.8 % grouped. Greedy samples cannot tell them apart; both read as fluent
    English.

    The expansion also has to happen inside each sequence: flattening (batch,
    head) first and then tiling pairs sequence 0's heads with sequence 1's data
    for any batch > 1.
    """
    batch, n_k, reps, hd = 3, 16, 3, 4
    n_v = n_k * reps
    q = torch.arange(batch * n_k * hd, dtype=torch.float32).reshape(batch, 1, n_k, hd)

    good = q.repeat(1, 1, reps, 1).reshape(batch * n_v, hd)
    for b in range(batch):
        for j in range(n_v):
            assert torch.equal(good[b * n_v + j], q[b, 0, j % n_k]), "tiling broke"

    grouped = q.repeat_interleave(reps, dim=2).reshape(batch * n_v, hd)
    assert not torch.equal(good, grouped), "tiling and grouping must differ"
    flattened_first = q.reshape(batch * n_k, hd).repeat(reps, 1)
    assert not torch.equal(good, flattened_first[: good.shape[0]]), "must tile per sequence"


def test_tiling_cannot_be_served_from_a_contiguous_head_shard() -> None:
    """Which is why the decode step gathers all K heads before selecting.

    Device d holds k-heads [d*n_k/D, ...) and v-heads [d*n_v/D, ...). Tiling
    sends global v-head j to global k-head j % n_k, and that walks straight out
    of the device's own block -- device 0's twelve v heads need k-heads 0-11,
    which live on three devices. No expansion of the four local heads produces
    it, so `TTModel.head_select` picks each device's twelve out of an
    all-gathered sixteen.
    """
    n_k, reps, n_dev = 16, 3, 4
    n_v = n_k * reps
    k_per, v_per = n_k // n_dev, n_v // n_dev
    for d in range(n_dev):
        needed = {(d * v_per + i) % n_k for i in range(v_per)}
        held = set(range(d * k_per, (d + 1) * k_per))
        assert not needed <= held, f"device {d} would not have needed the gather"
        assert len(needed) == v_per, "each device needs twelve distinct k heads"


def test_head_select_matrix_picks_the_right_heads() -> None:
    """The selection built in `TTModel.head_select`, as plain torch."""
    hd, n_k, n_dev = 4, 16, 4
    n_v_local = (n_k * 3) // n_dev
    gathered = torch.arange(n_k * hd, dtype=torch.float32)      # head h -> values h*hd..
    for d in range(n_dev):
        sel = torch.zeros(n_k * hd, n_v_local * hd)
        for i in range(n_v_local):
            src = (n_v_local * d + i) % n_k
            sel[src * hd : (src + 1) * hd, i * hd : (i + 1) * hd] = torch.eye(hd)
        out = (gathered @ sel).reshape(n_v_local, hd)
        for i in range(n_v_local):
            src = (n_v_local * d + i) % n_k
            assert torch.equal(out[i], gathered[src * hd : (src + 1) * hd]), (d, i)


def test_reinject_broadcast_equals_the_slice_form() -> None:
    """The broadcast re-inject must equal the slice/concat form.

    Valid only because the flattened hyper-connection layout is stream-major:
    stream c occupies columns [c*hidden, (c+1)*hidden). The device form takes
    `inject` channel-major as [.., hc, M, 1] and broadcasts it against
    [.., 1, M, hidden]; measured at 0.372 ms/call against 1.010 ms for the
    slice form and 9.634 ms for a repeat_interleave-based one.
    """
    torch.manual_seed(0)
    batch, hidden, hc = 3, 8, 4
    branch = torch.randn(1, 1, batch, hidden)
    inject = torch.randn(1, 1, batch, hc)
    hyper = torch.randn(1, 1, batch, hc * hidden)

    reference = hyper + (branch.unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2)
    slice_form = hyper + torch.cat([branch * inject[..., c : c + 1] for c in range(hc)], dim=-1)

    inject_c = inject.permute(0, 3, 2, 1)                       # [1, hc, M, 1]
    prod = (inject_c * branch).permute(0, 2, 1, 3).reshape(1, 1, batch, hc * hidden)
    broadcast_form = hyper + prod

    assert torch.equal(slice_form, reference)
    assert torch.equal(broadcast_form, reference)


def test_the_hyper_connection_mix_averages_in_one_op() -> None:
    """`mean` rather than `sum` then a scalar multiply, 97 times a step.

    Worth 29.6 ms eager (498.7 -> 469.1) and nothing traced, which is the useful
    part of the measurement: 0.30 ms per removed op is a host dispatch, and a
    trace replays with one. Fewer launches is an eager-path optimisation.
    """
    import inspect

    from twtest.tt.ops import gated_residual_mix

    src = inspect.getsource(gated_residual_mix)
    assert "ttnn.mean(per_stream, dim=-2, keepdim=True)" in src
    assert "ttnn.sum(per_stream" not in src
    assert "1.0 / hc_count" not in src.split("inject = None")[0]
