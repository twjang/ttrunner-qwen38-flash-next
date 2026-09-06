# Optimising batch-1 decode on Blackhole: principles, pitfalls, and the machine

This is the distilled companion to `HANDOFF.md`. That file is a chronological
log; this one is what a person needs to know *before* touching the decode path,
organised so it can be read once and kept.

Everything here is measured on this machine and this model unless it says
otherwise. Numbers are traced, batch 1, and quoted with the regime they were
taken in, because in this system that matters more than it usually does.

---

## 1. The machine

**Four Blackhole p150a, one mesh.** Per device:

| | |
|---|---|
| Usable Tensix grid | 11 x 10 = **110 cores** |
| L1 per core | **1572864 B** (1.5 MB) |
| Achievable DRAM bandwidth | **~388 GB/s** |
| Tile | 32 x 32; page size follows dtype (bfloat16 2048 B, float32 4096 B) |

### 1.1 The four devices are a ring, not a line

Probed by attempting `setup_fabric_connection` on all twelve ordered pairs:

    D0 -> [1, 2]     D1 -> [0, 3]     D2 -> [0, 3]     D3 -> [1, 2]

The links are **0-1, 0-2, 1-3, 2-3** — the cycle **0-1-3-2-0**. `D1-D2` and
`D0-D3` do not exist.

This matters because both `ttnn.MeshShape(1, 4)` and `Topology::Linear` read as
a line, and neither describes the wiring. Any multi-chip walk must follow the
ring; the natural `chip -> chip + 1` is wrong on two of its four steps and fails
with `Could not find any forwarding direction from (M0, D1) to (M0, D2)`.

### 1.2 What is sharded and what is replicated

Not obvious from shapes, and it changes what a cross-device comparison means:

* **DeltaNet v-heads are sharded**: `n_v_local = linear_num_v_heads // n_dev`,
  so each device holds *different* heads and `st.recurrent` legitimately differs
  device to device. A `ConcatMeshToTensor` of it returns 4 x 12 distinct heads,
  not four copies of twelve.
* **Dense weights are replicated**; the MoE expert stacks are sharded on the
  expert axis; `down_w` in the hyper-connection is `Shard.ROW`, so its matmul
  returns a *partial* sum and must be all-reduced before anything nonlinear
  touches it.

---

## 2. The model, and where its time goes

48 layers: **36 gated-DeltaNet + 12 QSA**. Hidden 2560, 512 experts with top-10
routing, expert intermediate 640, DeltaNet head dim 128.

### 2.1 There are two decode regimes, and they differ by 3x

`indexer_budget = 2048`. Below that position the sparse selection is *provably*
a no-op — at position p there are `p // 4` eligible blocks against a top-k of
512 — so `TTEngine` runs with `selection_active = False` and only flips it at
`max(positions) + 1 >= indexer_budget`.

Measured in one harness, one sequence length, changing only that flag:

| `selection_active` | median | tok/s |
|---|--:|--:|
| False (positions 0..2047) | **32.3 ms** | 31.0 |
| True (positions >= 2048) | **61.5 ms** | 16.3 |

**The selection costs ~29 ms a token**, and cost 73.5 until handoff 45.44. Every
conversation decodes its first 2048 tokens in the cheap regime and everything
after in the expensive one. A number quoted without its regime is meaningless
here.

The 44 ms that came out of it is this document's clearest example of 3.1, and of
a rule worth stating on its own: **`ttnn.topk` is O(nb x k)** -- 4.4 ms for
k=512 of nb=2048, 9.4 ms of 4096, linear in both, and `sorted=False` changes
nothing, so it is the *search* and not the sort. It was being run over every
block in the **allocated** context when only the first `p // ratio` are eligible
and the rest carry a -inf bias they can never win from. Before reaching for a
top-k, ask how many candidates can actually be returned.

### 2.2 Component map of the 32.65 ms step

Single-part ablation, `selection_active = False`:

| component | cost | note |
|---|--:|---|
| DeltaNet (36 layers) | **8.36 ms** | the recurrence is only ~2.95 of it |
| `gated_residual_mix` | **6.68 ms** | 96 invocations a token, ~7 ops each |
| MoE (48 layers) | **5.67 ms** | already on the wide-expert path |
| collectives | **3.86 ms** | 181 calls |
| QSA (12 layers) | **3.66 ms** | plus ~29 ms above once selecting |
| shared expert | 1.18 ms | |
| `reinject` | 0.04 ms | finished |
| PLE | 0.03 ms | finished |

Single-part costs overlap, so they sum to less than the whole; this is a
ranking, not a budget.

### 2.3 The budget, and why the roofline is a floor

    bytes          7.0 ms    2.73 GB a device at 388 GB/s
    dispatch       3.1 ms    2218 calls at the 1.4 us traced floor
    ---------------------------------------------------------
    accounted     10.1 ms
    remainder     22.5 ms

The remainder is located: **460 `ttnn.linear` calls carry 2.197 GB — 80 % of all
bytes moved — and would take 5.66 ms at full bandwidth.** They are most of the
step instead.

The weight read is ~3.17 GB a device a token, **8.2 ms at 388 GB/s = 122 tok/s**.
A perfect implementation — every GEMV at full bandwidth, zero launch cost, zero
collectives — still spends that 8.2 ms. It is a lower bound on time and **not a
lever**, which is the single most important thing to internalise before planning
work here.

---

## 3. The principles that actually decide things

### 3.1 At M = 1, a matmul's width is its parallelism

`ttnn.linear` gives each output tile to one core. A matmul whose output is `nt`
tiles therefore uses `nt` of the 110 cores and gets `nt / 110` of the bandwidth:

| weight | calls a token | output tiles | cores used |
|---|--:|--:|--:|
| `attn_qkv` [2560, 4608] | 36 | 144 | all 110 |
| [2560, 1536] | 72 | 48 | 48 |
| [2560, 1280] | 97 | 40 | 40 |
| [2560, 640] | 48 | 20 | 20 |
| router [2560, 512] | 84 | 16 | **16** |

This is the structural reason the step is four times its roofline, and it is why
"width is free until it fills the grid" is the right mental model rather than
"narrow is cheap".

### 3.2 Bytes are not the currency at M = 1

Measured directly, at M = 1:

| shape | us | MB | GB/s |
|---|--:|--:|--:|
| `attn_qkv` [2560, 4608] | 36.84 | 12.53 | 340 |
| `attn_gate` [2560, 1536] | 30.77 | 4.18 | 136 |
| shexp gate\|up replicated [2560, 1312] | 30.79 | 3.57 | 116 |
| shexp gate\|up sharded [2560, 352] | 29.54 | 0.96 | 32 |
| shexp down replicated [640, 2560] | 32.08 | 1.74 | 54 |
| shexp down sharded [160, 2560] | **38.23** | 0.44 | 11 |

**A thirteenfold cut in bytes buys 20 % of the time, and one of these shards is
slower than the tensor it replaces.** A `ttnn.linear` at M = 1 costs ~30 us
whatever its shape; only the very largest weights are anywhere near bandwidth.
Sharding narrows a shape, and a narrower shape sits further down the curve in
3.1, so the byte saving is spent on the efficiency it costs.

Corollary: arithmetic on bytes predicts nothing about time here. Several plans
built that way have had the wrong *sign*.

### 3.3 Only deep fusions pay

A chain costs about `max(bytes, launches x floor)`. Removing **one** launch is
nearly free — its cost overlaps with the neighbours that remain. The historical
record is unambiguous:

    delta scalars    8 launches a layer removed   +1.16 ms   (4.0 us each)
    causal conv     10 launches a layer removed   +1.58      (4.4 us each)
    qkv heads        9 launches a layer removed   +0.77      (2.4 us each)
    delta tail       5 launches a layer removed   +0.74      (4.1 us each)
    slice + silu     1 launch  a call  removed    +0.02      (0.2 us each)

Fuse eight ops or do not bother. And fuse for the right reason: **to stop
reading the same bytes twice, or to do less arithmetic — not to save
dispatches**, which are ~1.4 us in a trace.

Right-sizing *every* op in the step is worth only 3.94 ms by the census's own
model, and 0.04 ms of that on the 1000 widest calls. The small ops are not where
the time is.

### 3.4 Tracing is worth 17x

Same process, same sequence length, `selection_active = False`:

    eager    545.98 ms   (1.8 tok/s)
    traced    32.0  ms   (31.3 tok/s)

Anything measured eagerly is measuring a different program's cost structure.

---

## 4. Measurement discipline

These are not style preferences. Each one has cost this project a wrong
conclusion that survived until someone re-measured.

### 4.1 An isolated trace over-prices a small op chain by 2-4x

Thirty copies of one op cannot overlap; the model's neighbours can. Use isolated
timing to *rank* candidates, never to predict a saving. Decide on a paired
in-model A/B.

### 4.2 Measure both arms in one process

Assembling one arm's number from one harness and the other's from a second is
two experiments, not a comparison. The eager-vs-traced gap above reads as
"tracing buys nothing" if you take the two numbers from different scripts.

### 4.3 Know which regime and which flags you are measuring

Several model flags default to values the shipping engine does not use:

| flag | default | what ships |
|---|---|---|
| `TTModel.selection_active` | `True` | `False` below `indexer_budget` |
| `TTModel.trace_safe_rings` | `False` | `True` — only `TracedDecoder` sets it |
| `bench_step.py` `max_seq_len` | 262144 | the regime you care about |

Every one of these has produced a number that described a program the model
never runs. Print the regime with the result; the harnesses here now do.

### 4.4 Quote a rate with its sample size, and pool before believing a difference

The traced-capture hang has a base rate near 12 %. Read 8-of-8 as "zero" and the
next 6-of-8 looks like a regression; pooled, they are the same rate. Fisher on
8/8 against 6/8 is p = 0.47.

### 4.5 A guard that declines must say so

A fused entry point that silently returns `None` makes a measurement compare a
path with itself. Every one here warns once, with the reason. Set
`TT_STRICT_KERNELS` to turn declines into exceptions when a measurement depends
on the fused path actually running.

### 4.6 Gate correctness before believing a timing number

A kernel that is fast and silently wrong costs more than it saves. Score
`device_quality.py` **in both regimes** (4.3) before any timing is banked. And
for anything that writes back state it will read next step, check a *sequence*
of steps: per-step error compounds, and the one-shot number gives no warning.

---

## 5. Framework and hardware pitfalls

### 5.1 A trace charges circular buffers per program, not per execution

Programs run one at a time, so sizing a CB for the largest chunk it will ever
stream *looks* free. It is not: multiply it by every copy of that program in the
capture.

Worked example. One `ksgemv` program at the router shape reserves **81920 B a
core** (`cb0` and `cb1` are `cap x page`, i.e. the whole k-chunk of activation
and weight). Sixteen such programs is 1.31 MB of the 1572864 B L1 — 83 % —
leaving 260 KB for every other op in the same capture. Measured: four programs
run, sixteen hang. **Size streaming CBs for pipeline depth, not payload.**

### 5.2 Allocating device buffers inside a trace capture corrupts the replay

`Allocating device buffers is unsafe due to the existence of an active trace`.
Anything a fused path allocates lazily must be allocated before capture — and
note that Python evaluates arguments eagerly, so
`f(..., ttnn.transpose(k), make_output(...))` pays both even when the guard
inside `f` declines.

### 5.3 One unpacker per circular buffer

A compute kernel configures its unpacker from a CB. Reading a float32 tile
through a bfloat16-configured one gives garbage. Most `*_init` helpers take the
CBs they operate on and reconfigure; a bare `copy_tile` after an `init_sfpu`
that named a *different* CB does not.

### 5.4 A fused kernel's output dtype is part of its interface

Page size is derived from dtype and is a structural property of a kernel's CBs.
Widening an output dtype can trip a page-size guard in the same kernel, and can
make the *next* fused kernel downstream decline — turning a win into a loss.
Check what consumes an output before choosing its dtype.

### 5.5 `matmul_tiles` contracts the full 32

Using it as an outer product of a column against a row is only correct if the
operands' padding is zero, and nothing guarantees that for a tensor produced by
`typecast`, `transpose` or a slice. Zero the padding yourself, or use an op that
does not contract over it.

### 5.6 An ablation stub must return a *new* buffer

A stub that returns its input object hands the caller a tensor the captured
graph expects some op to have written, and the capture hangs — silently, so it
reads as device instability rather than a bug in the probe. `ttnn.clone(x)`.

### 5.7 Fabric, in six facts

1. The send functions live behind
   `using namespace tt::tt_fabric::linear::experimental`.
2. `ttnn.setup_fabric_connection` **mutates** the `ProgramDescriptor` (it
   appends semaphores) and returns exactly the argument block
   `WorkerToFabricEdmSender::build_from_args` consumes; splice it in at a fixed
   runtime-arg index.
3. `MeshProgramDescriptor` is keyed by `MeshCoordinateRange`, not
   `MeshCoordinate` (a bare coordinate is a `std::bad_cast`).
4. `NocUnicastAtomicIncFusedCommandHeader` takes four fields.
5. A never-taken `if constexpr` branch is still fully compiled.
6. The connection is opened to the **adjacent** node; distance is the send's
   `num_hops`. `FabricNodeId(chip + n)` appears to work at n = 1, 2 and dies at
   n = 3.

And two more that decide a design:

* A fabric send takes an **L1** source and a NOC address on a worker core — not
  a DRAM `buffer_address()`.
* **A `generic_op` with an identical CB list on every chip can address a peer's
  CB by its own pointer to it**, because the allocator puts that CB at the same
  L1 address everywhere. This removes the only thing that looked like it needed
  a cross-device address exchange, and it is what makes a multi-chip
  `generic_op` collective practical.

### 5.8 Hop distance is free

    launch only (control)   87.84 us
    one hop,   5120 B       83.57 us
    two hops,  5120 B       80.65 us
    three hops, 5120 B      75.29 us

All within a 12.6 us spread on an 88 us launch. A multi-chip reduce can be a
**chain**; a tree buys nothing.

---

## 6. What is closed, so nobody re-opens it

Each was closed by measurement, and the measurement is in `HANDOFF.md`:

* **byte reduction as a lever** — see 3.2;
* **the `attn_qkv` v-head regroup** — 0.079 ms, a tenth of its estimate;
* **sharding the shared expert** — +0.24 ms, the wrong sign;
* **extending the k-split to more call sites** — no measurable gain, and it
  destabilises the capture in proportion to how many programs are added;
* **merging collectives**, **replicating `hc_down`**, **`TT_GG_COLS=1`**.

## 7. What the target needs, honestly

The identified work supports **12-16 ms (60-80 tok/s)**. The arithmetic for
7.9 ms needs every narrow GEMV on a k-split *and* the collectives near zero
*and* the per-op floor across 2218 launches, and it still does not obviously
close. Getting past that needs an idea this document does not contain, and
saying so is more useful than another estimate built by adding up leads.
