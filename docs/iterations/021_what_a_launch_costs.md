# 021 — What a launch costs, measured on the path that pays for it

5.7 asks for fewer device launches and has asked for it since 012. Every
decision under it — three candidates priced, one kept and two refused — was made
against a per-dispatch cost of **0.30 ms** that came from a single A/B on a
different path. That constant was never checked against the path it was being
applied to, and it does not survive the check.

## Observation 1 — the cost model predicted four times the measurement

`op_count.py` prints, for a 128-token chunked prefill:

    total device calls in one 128-token chunked prefill: 11681
    at ~0.30 ms an eager dispatch: 3.50 s of dispatch

The chunk takes **925 ms**. A model that predicts 3.5 s of dispatch inside 0.9 s
of wall clock is not off at the margin, it is wrong in kind, and it had been
printed under every reading of this item.

Where 0.30 ms came from is recorded in 5.7 and is sound on its own terms: the
hyper-connection mix was changed to average in one op instead of summing and
scaling, removing 97 ops from an *eager decode step*, and the step went 498.7 ->
469.1 ms. 29.6 ms over 97 ops is 305 us apiece. The error is not the arithmetic,
it is the generalisation — 97 particular ops on one path became "a host
dispatch" everywhere. The same over-reach shows up if it is applied to the path
it was measured on: 6143 calls x 0.305 ms is 1873 ms for a step that takes 498.7.

## Observation 2 — measure the slope instead of quoting a constant

`dispatch_cost_check.py` wraps `gated_residual_mix`, which runs a known 97 times
per chunk, so each call issues `k` extra ops, and times the chunk across `k`.
The gradient of wall clock against added dispatches is the per-dispatch cost,
taken on the path in question.

| injected op | +0 | +194 | +388 | +776 | slope |
|---|---|---|---|---|---|
| 32x32 tile | 917.5 | 919.7 | 939.7 | 983.7 | **90.0 us** |
| 128x2560 activation | 915.6 | 922.0 | 942.4 | 990.2 | **100.0 us** |
| `reshape` | 914.7 | 913.8 | 915.4 | 915.9 | **2.1 us** |

Two things fall out, and the second is the one that matters.

**It is the launch, not the arithmetic.** 640 times the data costs 1.11x. An op
on this path is priced by the act of issuing it and essentially not at all by
what it computes, which is what "dispatch-bound" actually asserts and what had
been assumed rather than shown.

**Counted calls are not dispatches.** An injected `reshape` costs 2.1 us — it is
a host-side view. `reshape` is 1094 of the 11681 calls `op_count.py` reports, so
the total it prints overstates the dispatches, and the gap in Observation 1 is
partly its own units. This also retires "11681 calls, essentially all dispatch"
as a way of talking about the chunk: some fraction of that number never reaches
the device.

Repeatability, since a single sample has been wrong twice on this project: five
runs of the small-op arm gave 97.9, 90.2, 111.2, 75.0 and 90.0 us. Call it
90 +/- 15.

## Observation 3 — a real removal is worth less than an injected addition

The injection measures the *marginal* op. What 5.7 needs is the value of
*removing* an existing one, and there is exactly one clean removal on record to
calibrate against: 5.8 lifted `moe_chunk` from 32 to 128, which collapses the
MoE from four row-groups to one and takes `shared_expert` from 960 calls to 240
and `route` from 384 to 96. That is 1008 calls removed, and it bought
918.6 -> 861.7 ms.

**57 us per removed call, against 90 us per injected one.** So injection is an
upper bound and the realisable figure is about 60 % of it. Both numbers are the
same order, which is the useful part: the conversion factor for this item is
**tens of microseconds per launch**, not the 0.30 ms it had been using — between
five and six times smaller.

## What that does to the item

Re-priced at ~57 us, against a chunk of 925 ms:

| what | calls | worth |
|---|---|---|
| the largest single call site (`shared_expert`) | 576 | 33 ms, 3.6 % |
| every site in the by-caller top twelve, all of them | 3712 | 212 ms, 23 % |

The second row is not an opportunity — those calls are doing the model's work
and none of them is free — it is the ceiling on the whole category. And the
three candidates that were actually reachable are already priced and closed:
`moe_chunk` past 32 and the shared-expert hoist both buy their dispatches with
bf16 accuracy (5.8), and the shared q/k/v permute removed 360 calls and ran 30 %
slower.

So the honest state of 5.7 is not "keep grinding". It is that the flat by-caller
distribution — no site above 4.9 % — plus a launch worth ~57 us puts every
remaining individual reduction under 4 %, while the change that would take the
launch count out of the equation wholesale is capturing the chunk, and that
still needs the upstream fix in `ttnn_bug_report/`.

## The lesson

012 measured one thing correctly and 5.7 spent four iterations applying it to a
path it was never measured on. The tell was available the whole time and in the
tool's own output: `op_count.py` printed a dispatch estimate four times larger
than the wall clock it was meant to explain, on every run, and nobody read the
two numbers against each other.

A constant carried across a boundary is an assumption, and this one was cheap to
check — the harness that settled it is sixty lines and one afternoon. Where a
cost model is steering decisions, make it predict something already measured
before trusting it to price something new. That is 018's lesson at one more
level of remove: there, a mechanism was written up on a confounded control; here,
a measurement was written up and then quietly extrapolated past what it measured.
