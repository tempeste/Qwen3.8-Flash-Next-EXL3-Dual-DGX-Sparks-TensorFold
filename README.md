# Qwen3.8-Flash-Next on two DGX Sparks with TensorFold, EXL3 experts

Serve **Qwen3.8-Flash-Next** from two NVIDIA DGX Sparks (GB10, 128 GB each, linked by their ConnectX-7 ports) through
an OpenAI-compatible API, with [TensorFold](https://github.com/ashhart/TensorFold)'s Zig engine (`tensorfold-native`)
split across the two Sparks at `--tp 2`: one rank on each.

This is the serving setup of a deployment that serves coding agents around the clock: **24 streams at once, each with
a 409,600-token window** (YaRN factor 2 over the trained 262,144), an **NVFP4 KV cache**, a **45 GiB prompt cache** that
keeps conversation states between agent turns, **routed experts in EXL3** (turboderp's 2.68 bpw quant) on NVIDIA's FP4
checkpoint, MTP drafts, image input and tool calls. The engine changes it needs are a series of 34 patches on upstream
TensorFold; `scripts/prepare.sh` clones TensorFold at a pinned commit, applies them with `git apply` and builds the
image. The layout follows MiaAI-Lab's
[Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold),
whose scripts this recipe adapts ([`CREDITS.md`](CREDITS.md)).

- Checkpoint: [`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) at
  `b9ce18a6887e18c089fdea8509a33e63429a3300` (95.24 GiB)
- Routed experts: [`turboderp/Qwen3.8-Flash-Next-exl3`](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3),
  branch `2.68bpw_h6_ng8` at `d660113325e1eba20e760a9063580139bdbdfc22` (2.5 bpw, the edge layers 3.0); only the
  routed-expert tensors are used (~37.5 GiB), cut per rank
- Engine: TensorFold upstream `main` at `089a18dff289d4c58043c1957ad0d7fe73d814b3` plus [`patches/`](patches)
- API model id: `Qwen3.8-Flash-Next`, port 8000

**Status.** The patch series, the image recipe, the launch command and the engine settings are those of the deployment
the numbers below come from, and were checked against it without a Spark (see [What was checked](#what-was-checked)).
On 22 Nov the scripts were run end to end on a pair of Sparks from a fresh clone: `prepare.sh` built the image from
upstream TensorFold and these patches, checked the already-downloaded weights against the pinned sums, and `start.sh`
served replies byte-identical to the deployment's (3 of 3 greedy, 4 of 4 seeded sampled). The weights download was
exercised only for its first 256 MiB (it matched the pinned files byte for byte), and the experts-reuse step was
rechecked on the Sparks on 23 Nov after a revision. Report what breaks on other setups.

## Performance and accuracy

All figures below were measured on one pair of DGX Sparks in controlled runs (**lab**) or read from the deployment's
own logs under real agent traffic (**real**). **Every Spark ran with its GPU clock capped at 1,800 MHz**
(`nvidia-smi -lgc 300,1800`; about 1.79 GHz under load), chosen for boxes that run around the clock; `nvidia-smi`
reports a default of 2,418 MHz. This recipe does not set the cap, so an uncapped pair may differ, most likely in prompt
reading, which is bound by compute. Every before and after below ran on the same Sparks at the same cap.

### Where this work started

This work started from MiaAI-Lab's single-Spark recipe,
[Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/tree/20e15e41f2f514a0d287710f4665ee1b5009ba41):
one TensorFold engine on one Spark (TP=1). Patches 0002-0003 port its patches 0001-0017 ([`CREDITS.md`](CREDITS.md)).
On 22 November it was run on these Sparks as published (its image, TensorFold v1.1.4; NVIDIA's FP4 checkpoint; fp8 KV
cache; 5 streams × 262,144 tokens and a 4.2-4.7 GiB prompt cache a Spark), once on one Spark alone and once as one copy
on each Spark behind a relay that keeps a conversation on one copy. The benchmark is MiaAI-Lab's
[sparkDash](https://github.com/MiaAI-Lab/sparkDash): its decode test sends a short prose prompt (~80 tokens) and asks
for 400 tokens at temperature 0.7 with thinking on, so it is close to decode-only load, not agent traffic; its
prompt-reading test sends one prompt at a time.

The single-Spark setup serves **10 streams × 262,144 tokens for the pair** (5 a Spark); this recipe serves **24 streams
× 409,600 tokens** across both Sparks, with a 45 GiB prompt cache.

| | streams × window | date, runs | decode 1 | 2 | 4 | 8 | 16 | prompt 4K | 32K | 131K | 262K |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TP=1 recipe, one Spark | 5 × 262,144 | 22 Nov, mean of 3 | 46 | 70 | 101 | 107 | 114 | 2,327 | 2,136 | 1,951 | does not fit |
| TP=1 recipe × 2, one copy a Spark | 5 × 262,144 a copy, 10 × 262,144 in all | 22 Nov, mean of 3 | 49 | 96 | 150 | 209 | 213 | 2,315 | 2,129 | 1,948 | does not fit |
| our engine at TP=1, one Spark | 12 × 409,600 | 22 Nov, mean of 3 | 49 | 82 | 136 | 208 | 220 | 2,225 | 2,105 | 2,034 | 1,976 |
| our engine at TP=1 × 2, one copy a Spark | 12 × 409,600 a copy, 24 × 409,600 in all | 22 Nov, mean of 3 | 52 | 102 | 169 | 254 | 401 | 2,237 | 2,106 | 2,037 | 1,974 |
| this recipe, engine of 15 Nov | 24 × 409,600 | 15 Nov, 1 run | 78 | 101 | 171 | 260 | 379 | 4,147 | 3,830 | 3,714 | 3,617 |
| this recipe, engine of 20 Nov | 24 × 409,600 | decode 20 Nov, median of 3; prompt 21 Nov, mean of 3 | 75 | 105 | 171 | 258 | 375 | 3,696 | 3,621 | 3,544 | 3,458 |
| this recipe, through `patches/0034` | 24 × 409,600 | 22 Nov, mean of 9 | 65 | 110 | 171 | 269 | 378 | 3,893 | 3,803 | 3,711 | 3,612 |

Tokens a second on sparkDash, decode for all streams together (lab). Dates are UTC.

- **Our engine at TP=1** is this recipe's engine and image on one Spark, without the settings that need two ranks:
  decode beside a prompt read, beside packing, overlap, mixed last, mixed sequence-parallel, ride fast, head sharding
  and the two-rank `TF_FLASHNEXT_TP_*` settings are off, because the engine refuses them at TP=1. Reply keep stays set
  but does nothing at TP=1. Everything else is as in production: late skip, prompt-cache row sharing, EXL3 experts
  (cut for one rank), expert re-route and the shared pool. Each copy serves 12 × 409,600 with a 23 GiB prompt cache,
  production's 45 GiB split in two.
- **TP=2 against our engine at TP=1 × 2.** On the same two Sparks, TP=2 reads a prompt 76-83% faster than TP=1 × 2 at
  4K-262K. Its total decode at 16 streams is about 5% lower (378 against 401), and at 1-8 streams the difference is
  within the noise. TP=2 is still the recipe, for three reasons:
  - Both Sparks read every prompt, and agents send 4.4-5.6 new prompt tokens for every token they generate
    ([Agent traffic](#agent-traffic-lab-and-real)).
  - Decode beside a prompt read needs two ranks, and on agent traffic it is what keeps decode going while prompts are
    read (the "Decode beside a prompt read" row below).
  - A single conversation can use the whole pair's memory: the 45 GiB prompt cache is one cache for every
    conversation, where a TP=1 copy keeps 23 GiB for the conversations the relay sends it.
- **Queueing.** The TP=1 recipe holds 5 streams on one Spark and 10 on the pair, and our engine at TP=1 holds 12 a
  copy. Requests past those counts wait at the engine or the relay: the TP=1 recipe's at 8 and 16 streams on one Spark
  and at 16 on the pair, our engine at TP=1's at 16 streams on one Spark.
- **262K** does not fit the TP=1 recipe's 262,144-token window with the template and the reply (sparkDash's own
  "Skipped" row).
- **Run counts.** Every row measured on 21 or 22 November is a mean over runs that each started a fresh engine, with
  the arms interleaved: 3 runs for each TP=1 row and for the 20 November engine's prompt reading (21 November), and 9
  for the engine through `patches/0034`. The 15 November row is one run; within a run, sparkDash's prompt reading is the mean of 3 samples at
  4K and 32K, 2 at 131K and 1 at 262K. Its 4K figure was high for its build: five later runs of the same build averaged
  3,944.
- The 15 November engine predates beside packing, reply keep and the main stream's priority with beside on. The 20
  November engine carries everything through `patches/0033` (five experts off).

### From the single-Spark recipe to this one

The changes in build order, with what each gained when it was measured. Each gain is that change against the engine
just before it, on the measure named; the measures differ from row to row, so the gains do not add up or chain.
Gains measured on 22 November compare production with production less that one change (for two ranks, with our
engine at TP=1 × 2 on the same two Sparks), 3 fresh-start runs a side, interleaved. A `*` after a gain means its 95%
interval includes no change. "Our open test" is one of the prompt-reuse tests in [`bench/`](bench), which anyone can
rerun; the rest are sparkDash or our own benchmarks named in the row.

| change (date) | gain | measure | lab or real |
|---|---|---|---|
| **Start:** the single-Spark recipe, one copy a Spark (as published; measured 22 Nov) | where this started: decode 49 / 96 / 150 / 209 / 213 at 1-16 streams for the pair (5 × 262,144 a copy, 10 × 262,144 in all), 46 / 70 / 101 / 107 / 114 on one Spark (5 × 262,144); prompt reading 2,327 / 2,136 / 1,951 at 4K / 32K / 131K, 262K does not fit its 262,144-token window | sparkDash decode and prompt reading, 3 fresh-start runs of each, interleaved (22 Nov; table above) | lab |
| Two ranks at TP=2, one a Spark (upstream PR #1784's head split; 2 Nov) | decode +19.1%* / +15.3%* / -1.6%* / +3.2%* / -5.4% at 1-16 streams; prompt reading +76.4% / +80.7% / +82.0% / +83.0% at 4K-262K; streams: our engine at TP=1 × 2 serves 12 × 409,600 a copy, 24 × 409,600 in all; TP=2 serves 24 × 409,600 | sparkDash decode and prompt reading, 3 fresh-start runs of each arm against 3 of production, interleaved (22 Nov); TP=1 × 2 is our engine at TP=1, one copy a Spark behind a relay that keeps a conversation on one copy, against TP=2 on the same two Sparks | lab |
| NVFP4 KV cache (engine 2 Nov; in production from 3 Nov) | decode -12.2%* / +5.9%* / +3.4%* / +2.9%* / +2.5%* at 1-16 streams; prompt reading -3.2%* / -0.8% / -0.9% / -0.7% at 4K-262K: no speed change on sparkDash; capacity: fp8 KV also served 24 × 409,600, so these tests did not show a capacity gain | sparkDash decode and prompt reading, 3 fresh-start runs of each arm against 3 of production, interleaved (22 Nov); the arm is production with fp8 KV, the format before it | lab |
| EXL3 routed experts on the FP4 checkpoint (engine 5 Nov; in production from 5 Nov) | decode -2.3%* / 0.0% / +11.2%* / +13.6%* / +12.7% at 1-16 streams; prompt reading -1.9%* / +3.3% / +3.4% / +3.3% at 4K-262K | sparkDash decode and prompt reading, 3 fresh-start runs of each arm against 3 of production, interleaved (22 Nov); the arm is production with the checkpoint's own NVFP4 experts | lab |
| Keep and frame-ordering settings for the TP=2 prompt path (8 Nov) | decode +18.8%, prompt reading +12.9%, token-exact | an agent-traffic benchmark | lab |
| Prompt-visit time budget (8 Nov) | worst decode stall cut by a third | longest decode stall | lab |
| Draft-head prompt idle fixes (8 Nov) | 12% more agent turns finished, equal quality | agent turns finished in a fixed time | lab |
| Larger RoCE messages, one attention launch a round, batched draft picks (8 Nov) | 28-stream decode round 172 → 146 ms, equal tokens | decode round time | lab |
| Batched verify picks (9 Nov) | decode +9% beside a long prompt, token-exact | decode beside a long prompt | lab |
| Decode HC split (10 Nov) | decode +6.5% at 27 streams, token-exact | decode-only | lab |
| Exact sequence-parallel sums (10 Nov) | prompt reading +1.65%, first replies ~2% sooner | prompt reading | lab |
| Mixed rounds: a prompt piece rides a decode round (10 Nov) | prompt reading while streams fill +21.6%, decode a stream +8.2% | mixed prompt and decode load | lab |
| Riding-piece fix (10 Nov) | decode +7.2%, prompt reading +7.1% at 28 streams | mixed load | lab |
| Four prompt-front settings (11 Nov) | cold prompt reading +9.5%, first-token time p50 −18.9% | mixed load | lab |
| Five DeltaNet and prompt-sum settings (11 Nov) | full-load decode and prompt reading +7.5%, stream gaps over 1 s −66% | full load | lab |
| Newer CUDA user-mode driver from the image (12 Nov) | decode a round +8.3% at light load; on the deployment +7.9% to +8.7% decode rounds a second a stream | light-load decode (lab); production hours before and after (real, observational) | lab and real |
| Picture prompts at text speed (12 Nov) | a picture prompt reads in 1.02x the time of text (was 1.7x) | picture-prompt reading | lab |
| 409,600-token window, kept pictures not encoded again (12 Nov) | warm first token at 20 pictures 16.5 → 1.56 s; busy decode 169.3 against 172.4 (1 run each) | picture conversations | lab |
| Expert re-route (13 Nov; changes replies) | speed alone: decode -4.6%* / -2.6%* / +0.1%* / +4.3%* / +6.0% at 1-16 streams, prompt reading -2.1%* / -0.1%* / 0.0% / 0.0% at 4K-262K (sparkDash, 3 runs each, 22 Nov); the busier rank reads 33.5% fewer expert bytes at 17-32 rows; quality 383 against 382 of 421 | expert bytes a round; a 421-task quality evaluation | lab |
| Prompt cache 30 → 45 GiB (13 Nov) | 45 GiB: the four oldest long documents back warm 12 of 12, first token 1.64 s; 30 GiB: 12 of 12, first token 1.64 s. The probe's kept documents still fit in 30 GiB, so both sizes brought all of them back warm; this test cannot show the larger cache's benefit | our open test, `bench/` (`reuseprobe.py ret`): nine LCQ documents of 262k-400k tokens (about 3.2M) read once, then the four oldest asked again; 3 fresh-start runs an arm, interleaved (22 Nov) | lab |
| Prompt-cache row sharing (14 Nov) | cold re-reads 18 of 24 → 10 of 73 returning conversations; first-token p50 68 → 4.9 s; speed +0.2% (within noise); same document, new question at 131k-262k: 24 of 24 resumed, first token 0.35 s with it; 24 of 24 resumed, first token 1.55 s with sharing and shared-prefix keep off (3 runs each, 22 Nov) | a prompt-cache replay; the new-question part is our open test, `bench/` (`reuseprobe.py warm`) | lab |
| Decode beside a prompt read: long prompt pieces on their own stream (14 Nov; fixed 16 Nov) | whole pair +27.8% / +24.8% / +13.7% at 8 / 16 / 24 streams; decode a stream beside a read +82% / +121% / +123%; prompt rows a second −32% / −32% / −28% | replayed agent traffic, 3 interleaved repeats | lab |
| Late skip: short drafted windows skip DeltaNet end-state writes (16 Nov; idea from Picom-code) | +1.5% to +2.1% at 8-24 streams, within noise; exact | decode-only, 1 run a level | lab |
| Any-size packing (relaxed; 18 Nov) | whole pair +1.6% and new prompt rows a second +2.6% at 16 streams; quality pass; no pack was logged, so the gain is not shown to come from packing | replayed agent traffic, 4 runs against 3 | lab |
| Shared-prefix keep (19 Nov) | workers that parted after a ~22.8k shared opening resumed it: 24 of 24, first token 0.79 s with it; 24 of 24, first token 1.22 s without it | our open test, `bench/` (`reuseprobe.py fork`): two sets of six requests sharing a ~22.8k-token opening, each then its own ~2.7k body; workers 3-6 of each set counted; 3 fresh-start runs an arm, interleaved (22 Nov) | lab |
| Reply keep (relaxed; 20 Nov) | on follow-up turns: rows read a follow-up 11,827 → 1,611, first-token p50 6.85 → 1.10 s; on the deployment, 542,927 prompt rows not read again in its first 3 hours. sparkDash sends no follow-up turns, so its decode test does not show this gain: +15.5% / -7.5%* / +1.1%* / +2.1%* / +1.1%* at 1-16 streams (3 runs each, 22 Nov) | a follow-up workload (lab); the engine's counters (real); sparkDash decode (lab) | lab and real |
| Stream-priority fix with beside on (21 Nov) | lone prompts read +4.4% to +5.8% faster at 4K-262K; replies byte-equal | sparkDash prompt reading, 3 fresh-start runs a build, alternated | lab |

Except for the gains measured on 22 November, the changes from 8 November on were measured on one pair against the
engine it then served; re-route, prompt-cache row sharing, reply keep and the priority fix were first rolled to one of
the deployment's two pairs.

### Decode (lab)

Whole-pair decode on sparkDash, tokens a second, with the whole series through `patches/0034`, each run after a fresh
engine start:

| streams | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| **this recipe** (EXL3 experts, 24 × 409,600; mean of 9, 22 Nov) | 65 | 110 | 171 | 269 | **378** |

- With the checkpoint's own NVFP4 experts in place of EXL3 (`EXL3_EXPERTS=0`): 61 / 107 / 157 / 238 / 336, against
  60 / 107 / 174 / 271 / 379 for production in the same interleaved rounds on 22 November, 3 runs each (the EXL3 row
  of [From the single-Spark recipe to this one](#from-the-single-spark-recipe-to-this-one)).
- Earlier engines on the same benchmark: the engine as it served on 15 November, one run a level, made
  78 / 101 / 171 / 260 / 379 (60 / 94 / 162 / 232 / 335 with NVFP4 experts); it predates beside packing, reply-keep
  and the main stream's priority with beside on. The engine of 20 November made 75 / 105 / 171 / 258 / 375, medians
  of 3 ([Five experts](#five-experts-off-by-default)).

### Prompt reading (lab)

One request at a time, tokens a second, with the whole series through `patches/0034`, each run after a fresh engine
start:

| | 4K | 32K | 131K | 262K |
|---|---:|---:|---:|---:|
| this recipe (mean of 9, 22 Nov) | 3,893 | 3,803 | 3,711 | 3,612 |

The nine runs stayed within 1% of their mean at 32K-262K; at 4K they ranged from 3,621 to 4,024. On 21 November, in 3
runs a build alternated, the engine before 0034 read 3,696 / 3,621 / 3,544 / 3,458 and the series through 0034 read
3,910 / 3,806 / 3,709 / 3,612 (the stream-priority row above).

### Agent traffic (lab and real)

Decode-only load and agent traffic are far apart. Agents send 4.4-5.6 new prompt tokens for every token they generate
(real), and on a pair whose GPU is already ~96% busy every prompt row read beside decode costs decode time.

| | 8 streams | 16 streams | 24 streams |
|---|---:|---:|---:|
| decode-only, whole pair, tok/s (lab) | ~250 | ~372 | ~449 |
| replayed agent-shaped streams, whole pair, tok/s (lab) | 144 | 182 | 180 |
| time to first token p50, agent-shaped (lab) | 3.4 s | 7.5 s | 18 s |

On real traffic the deployment's goal, 300+ generated tokens a second for the pair in 90% of busy minutes, **was not
reached**. In the busiest hour sampled (20 November), with up to ~10 streams a pair, the busiest minutes reached 290 and
253 tok/s on the two pairs, and no minute reached 300 (real).

### Long-context accuracy (lab)

Retrieval and copy tasks hidden in long synthetic documents, one request at a time, graded by exact match: 64 tasks a
depth from 4 documents at 131K-400K, 32 tasks from 2 documents at 8K-95K, temperature 0, Wilson 95% intervals; engines
compared task by task (McNemar and sign tests, Holm-adjusted across depths).

| depth | 8K | 32K | 65K | 95K | 131K | 200K | 262K | 330K | 400K | pooled 131K-400K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| this recipe | 32/32 | 32/32 | 31/32 | 32/32 | 63/64 | 58/64 | 63/64 | 62/64 | 46/48 | **0.961** [0.932, 0.977] |

At 400K the run reached 3 of its 4 documents before its deadline, so it has 48 tasks there. Repeats at temperature 0.7
(64 tasks at 200K and 400K, two repeats) scored 0.938 and 0.938, against 0.953 at temperature 0.

**The prompt cache resumes new questions on a document already read.** In an earlier one-pass run of this test, every
new question at 100K+ tokens on a document the engine had already read resumed from its kept state, 75 of 75, instead
of reading the document again.

### Five experts (off by default)

`TF_FLASHNEXT_EXPERTS_TOPK=K` (1-9; unset, `0` or `10`: off) keeps each token's top K of the model's 10 routed experts
and renormalises their weights over the K, in every MoE layer: decode, prompt reading and the draft head. With it off,
output is token-for-token identical to the engine without the switch. The engine refuses it together with expert
re-route, so `start.sh` leaves out `TF_FLASHNEXT_EXPERT_REROUTE` when it is set. **It is faster and lower quality**
(lab, three interleaved runs a side, medians, 20 November):

| sparkDash, whole-pair decode tok/s | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| ten experts (the default) | 75 | 105 | 171 | 258 | 375 |
| five experts | 80 | 118 | 179 | 274 | 378 |
| change | +6.0% | +12.4% | +4.8% | +6.5% | +0.9% |

| long-context accuracy | 8K | 32K | 65K | 95K | 131K | 200K | 262K | 330K | 400K | pooled 131K-400K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ten experts | 32/32 | 32/32 | 31/32 | 32/32 | 63/64 | 58/64 | 63/64 | 62/64 | 46/48 | **0.961** [0.932, 0.977] |
| five experts | 30/32 | 30/32 | 25/32 | 26/32 | 54/64 | 47/64 | 42/64 | 51/64 | 40/64 | **0.731** [0.680, 0.777] |

- Paired by task on the 304 tasks both ran at 131K-400K: 292 against 225, a 22-point loss (Holm-adjusted p ~5e-13);
  per depth the loss is significant at 262K, 330K and 400K. Temperature 0.7 repeats agree (0.625 and 0.656).
- A replay of 48 agent tool turns at temperature 0 failed the quality bar: median similarity to the ten-expert replies
  0.461, the same first tool on 78.7% of turns, replies a median 34% longer; tool calls stayed valid.
- Prompt reading was 1.3-3.9% slower; replayed agent-shaped load was +8.6% / +8.0% at 8 / 16 streams; draft acceptance
  fell from 0.632 to 0.607.
- At 16 decode-only streams the gain is ~0 because expert re-route (on by default) already reads about the same set of
  experts there.

Turn it on with `TF_FLASHNEXT_EXPERTS_TOPK=5 ./start.sh restart` (or `export TF_FLASHNEXT_EXPERTS_TOPK=5` in
`scripts/local.sh`).

## Requirements

- **Two DGX Sparks** (or two GB10 systems with 128 GB unified memory) with nothing else large on their GPUs. `start.sh`
  refuses a Spark with less than `MEM_NEED_GIB` (100) GiB `MemAvailable`; that number is a guard chosen by this recipe,
  not a measurement of what a rank needs (`MEM_CHECK=0` makes it a warning).
- **A direct ConnectX-7 link:** a QSFP cable between the CX7 ports (200 Gb/s), an IPv4 address on each end in one
  private subnet (`ping` must work) and a RoCE v2 GID on each. Both PCIe twins of the cabled port are used as NCCL rails
  when both are up and addressed (`NCCL_RAILS=1`: one).
- **Key-based ssh** from the first Spark (the head: rank 0 and the API) to the second (the worker):
  `ssh -o BatchMode=yes user@<worker> true` must work.
- On both: Docker with the NVIDIA container runtime (your user in the `docker` group), `rsync`, `sha256sum` and
  `python3`; on the head also `curl` and the `hf` CLI or Python's `huggingface_hub`.
- **Disk.** On each Spark ~96 GiB for the checkpoint under `~/.cache/huggingface`, the EXL3 experts of its rank
  (~19 GiB on the head, ~18 GiB on the worker; the head holds both ranks', ~37.5 GiB, until the worker has its copy),
  and the image under Docker's root (`IMAGE_FREE_GB`, 40 GB asked for; the build's own size is not measured here).
  `EXL3_SOURCE=download` also keeps the six EXL3 shards (~43.2 GiB) on the head.
- Network on the head for the first run: GitHub (TensorFold, libjpeg-turbo, TensorFold's v1.1.4 release),
  ziglang.org, storage.googleapis.com (libwebp), nvcr.io (the base image) and Hugging Face.

## Quick start

On the head:

```bash
git clone https://github.com/tempeste/Qwen3.8-Flash-Next-EXL3-Dual-DGX-Sparks-TensorFold.git
cd Qwen3.8-Flash-Next-EXL3-Dual-DGX-Sparks-TensorFold
cp scripts/local.sh.example scripts/local.sh     # set WORKER=user@<worker> (and FABRIC_PEER if needed)
./start.sh
```

The first run builds the image (about 15 minutes), copies it to the worker, downloads and checks the checkpoint and the
EXL3 experts, and copies the worker's share over the link. Later starts load the weights and serve. Then:

```bash
curl -s http://<head-address>:8000/v1/models
curl -s http://<head-address>:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-Flash-Next",
  "messages": [{"role": "user", "content": "Write a Python fibonacci function."}],
  "max_tokens": 2000
}'
./start.sh restart        # restart both ranks, e.g. after changing a setting
./stop.sh                 # stop both ranks (each rank's log is saved first)
docker logs -f qwen38-flash-next-tf               # rank 0's log
ssh <worker> docker logs -f qwen38-flash-next-tf  # rank 1's log
```

The model thinks before it answers (`reasoning_content`) at `reasoning_effort` `xhigh` by default, so give replies
enough `max_tokens` (32,768 when a request sets none).

## What `start.sh` and `scripts/prepare.sh` do

**`./start.sh`**:

1. **Settings:** checks them, and builds the engine's variables from `scripts/engine.env` (yours win).
2. **Setup:** runs `scripts/prepare.sh` when what it last prepared differs (another image, checkpoint, experts or
   worker); `PREPARE=1` forces it, `PREPARE=0` skips it.
3. **Checks:** the image, the checkpoint and the experts on both Sparks; the link (the route to the worker, its CX7
   netdev, RoCE devices and GID); a running server (stopped on `restart`); a stopped container of the same name
   (removed, its log saved first); the port on both; `MemAvailable` on both.
4. **Launch:** rank 1 on the worker over ssh, then rank 0 here:
   `tensorfold-native serve <snapshot> --tp 2 --master <head's link address> --master-port 29500 --name
   Qwen3.8-Flash-Next --context 409600 --parallel 24 --kv-cache nvfp4 --prompt-cache-gib 45 --max-tokens 32768
   --temperature 1.0 --top-p 0.95 --top-k 20 --reasoning-effort xhigh --no-update-check --thinking --vision ...
   --rank R --host H --port 8000`, with NCCL on the link's RoCE devices, the 105 engine variables, and the image's
   CUDA forward-compatibility library first on the library path (`CUDA_COMPAT=1`). Rank 1 binds 127.0.0.1 and serves
   no API.
5. **Wait and smoke test:** `/health`, then one greedy chat completion with thinking off.

`DRY_RUN=1 ./start.sh` prints both ranks' `docker run` commands and starts nothing. Extra arguments go to both ranks
after the defaults (`./start.sh restart --max-tokens 16384`).

**`scripts/prepare.sh`** (safe to re-run; each step skips work already done):

1. Preflight on both Sparks: Docker, the GPU, `rsync`, key-based ssh, the RoCE link, disk.
2. The image `tensorfold-qwen38-tp2:089a18dff-<hash>`: TensorFold cloned at `TF_REF`, every `patches/*.patch` applied
   with `git apply` in filename order, built with Zig 0.17.0 (sha256-checked) on `nvcr.io/nvidia/pytorch:26.07-py3`:
   `zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=fast fatbins native`. The serving image also carries
   libjpeg-turbo 3.1.4.1 and libwebp 1.6.0 (image decoding, built from sha256-checked sources) and the grammar library
   of TensorFold's v1.1.4 release (sha256-checked); NOTICE lists their licenses. The CUDA kernels are compiled into the
   build (no separate kernel set).
3. The same image on the worker (`docker save | docker load`), checked identical.
4. The checkpoint at its pin, into `~/.cache/huggingface`, checked file by file against `scripts/checkpoint.sha256`
   (the Hub's sha256 of every large file).
5. The EXL3 experts into `~/.cache/huggingface/exl3-experts-2.68bpw` by `tools/exl3_experts_pack.py` (below), each
   output file checked against `scripts/exl3-pack.sha256`. A folder there (`EXL3_PACK`) that already holds
   `manifest.json` and, with those sums, the index and rank 0's files is used as it is, and nothing is fetched; while
   the worker lacks rank 1's files, the head's folder must hold rank 1's too (it copies them), or they are fetched
   again.
6. The worker's copy: the checkpoint over the link with `rsync` (then checked against `scripts/checkpoint.sha256`
   there), and rank 1's expert files with the index and `manifest.json` (and `SHA256SUMS` when the head has one),
   checked against `scripts/exl3-pack.sha256` there. A worker folder that already holds them with the pinned sums is
   not copied to. Rank 1's files are then removed from the head (`EXL3_PRUNE=0` keeps them).

The ready markers (`.ok-head`, `.ok-worker` in the experts folders) are trusted only while every pinned file they cover
is there at its pinned size; otherwise the folder is checked again. The fetch refuses to write into an output file that
is a symlink or has another hard link: remove the link or give a fresh `EXL3_PACK` folder.

`scripts/prepare.sh --check-patches [DIR]` clones TensorFold at `TF_REF` and applies the patches without docker.

### The EXL3 experts

`TF_FLASHNEXT_EXPERTS_EXL3=DIR` makes the engine load the routed experts from an experts-only EXL3 folder and
everything else from the FP4 checkpoint. The folder holds safetensors files under the EXL3 checkpoint's own tensor names
(`model.language_model.layers.L.mlp.experts.E.{gate,up,down}_proj.{trellis,suh,svh,mul1}` and the MTP head's
`mtp.layers.0.mlp.experts.E...`) and one `model.safetensors.index.json`. Rank 0 holds experts 0-255 of every decoder
layer and the MTP head's experts, rank 1 experts 256-511; the engine reads the index, skips files that are not on its
Spark, and loads the experts its rank owns.

`tools/exl3_experts_pack.py` makes that folder from the pinned branch:

- `EXL3_SOURCE=range` (default): reads the six shards' headers, then fetches only the routed-expert byte ranges with
  `curl -r` (~37.5 GiB instead of the whole branch), resumable.
- `EXL3_SOURCE=download`: downloads the six shards (~43.2 GiB), checks each against the Hub's sha256, and cuts them.

Both write the same files: the shards' headers are checked against pinned sums first, and every output file, and the
index, against the sums of the folder the deployment serves from (`scripts/exl3-pack.sha256`). The tool only parses
safetensors JSON headers and copies byte ranges; nothing from the checkpoint is run.

## Configuration

Every setting lives in [`scripts/config.sh`](scripts/config.sh); set one for a run from the environment
(`PARALLEL=16 ./start.sh restart`), or keep it in `scripts/local.sh` (sourced as bash) or `.env` (plain `KEY=value`
lines, read, never run). The first that sets a value wins: the environment, `scripts/local.sh`, `.env`, the default.

| Variable | Default | Meaning |
|---|---|---|
| `WORKER` / `FABRIC_PEER` | empty | the worker's ssh target, and its CX7 address when `WORKER` is on another network |
| `MASTER_ADDR` / `MASTER_PORT` | the head's link address / `29500` | the ranks' rendezvous (keep it on the private link) |
| `CONTEXT` | `409600` | prompt + reply window per stream; past 262,144 it needs `TF_FLASHNEXT_YARN` (2: up to 524,288) |
| `PARALLEL` | `24` | streams decoded together |
| `KV_DTYPE` | `nvfp4` | `--kv-cache`: `nvfp4` or `fp8` (lossy), `bf16` (exact) |
| `PROMPT_CACHE_GIB` | `45` | memory for kept conversation states (prefix reuse) |
| `MAX_TOKENS` / `THINKING` / `REASONING_EFFORT` | `32768` / `1` / `xhigh` | a request's defaults |
| `TEMPERATURE` / `TOP_P` / `TOP_K` | `1.0` / `0.95` / `20` | sampling defaults |
| `DRAFTS` | `1` | MTP drafts (`0`: `--no-drafts`) |
| `VISION`, `MAX_IMAGES`, `MAX_VIDEOS`, `IMAGE_TOKENS` | `1`, `50`, `4`, `16384` | image and video input and its budgets |
| `SERVED_NAME` / `HOST` / `PORT` | `Qwen3.8-Flash-Next` / `0.0.0.0` / `8000` | the API model id and where rank 0 listens |
| `EXL3_EXPERTS` | `1` | EXL3 routed experts; `0`: the checkpoint's own NVFP4 experts |
| `EXL3_SOURCE` / `EXL3_LIMIT_RATE` | `range` / `0` | how the experts are fetched ([above](#the-exl3-experts)); a curl rate limit |
| `CUDA_COMPAT` | `1` | the image's CUDA forward-compatibility library ahead of the host's (`0`: the host's) |
| `ENGINE_ENV_FILE` / `ENGINE_UNSET` | `scripts/engine.env` / empty | the engine's variables; names to leave out |
| `MEM_NEED_GIB` / `MEM_CHECK` | `100` / `1` | the `MemAvailable` guard before launch |
| `PREPARE`, `WAIT_TIMEOUT`, `STOP_TIMEOUT`, `LOG_DIR`, `LOG_KEEP`, `DOCKER_RUN_EXTRA` | | setup and operation |

Less common: `MODEL_ID` / `MODEL_REVISION`, `EXL3_REPO` / `EXL3_REVISION` / `EXL3_PACK`, `TF_REPO` / `TF_REF`,
`BASE_IMAGE`, `ZIG_VERSION`, the codec and release pins, `IMAGE`, `CONTAINER_NAME`, `HF_CACHE`, `WORKER_HF_CACHE`,
`KERNEL_CACHE`, `STATE_DIR`. A changed pin needs matching sums in `scripts/checkpoint.sha256` or
`scripts/exl3-pack.sha256`.

### The engine's variables

[`scripts/engine.env`](scripts/engine.env) lists every `TENSORFOLD_*`, `TF_FLASHNEXT_*` and `TF_EXL3_*` variable the
deployment serves with (and `NCCL_MAX_NCHANNELS=4`), grouped by the patch that adds each; each patch's message
describes its flags ("Flags this patch adds"). `start.sh` passes them to both ranks, which compare their settings when
they connect. Any of these names set in the environment, `scripts/local.sh` (exported) or `.env` replaces the file's
value; `ENGINE_UNSET="A B"` leaves names out. `TENSORFOLD_API_KEY` (the server's API keys) goes to rank 0 by name
only. With the default `HOST=0.0.0.0` and no key, anyone who can reach port 8000 can use the model: set a key, or
`HOST=127.0.0.1` behind your own gateway.

What the defaults are, in short:

- Most of the series is exact: with a flag on, the tokens are the same as without it, and several flags exist only
  to make the two ranks faster at the same bits.
- Some paths are relaxed rather than bit-exact and were accepted on quality evaluations against a reordering-noise
  floor: `TF_FLASHNEXT_TP_PROMPT_RELAXED`, `TF_FLASHNEXT_GDN_CHUNK_RELAXED`, `TF_FLASHNEXT_TP_SCATTER_BF16`,
  `TF_FLASHNEXT_MIXED_BESIDE_PACK_ANY` and `TF_FLASHNEXT_REPLY_KEEP` (a finished reply keeps its lane's state, so the
  agent's next turn does not read the reply back as prompt).
- `TF_FLASHNEXT_EXPERT_REROUTE=5,24,8` (expert re-route) changes replies: from 8 streams, each row keeps its top 5
  experts and fills its other 5 slots with experts other rows already picked, within its own top 24, so a round reads
  fewer experts. On a 421-task quality evaluation the two settings scored 383 and 382 (lab).
  `ENGINE_UNSET=TF_FLASHNEXT_EXPERT_REROUTE` turns it off.
- Five of the variables have no reader in the patched tree and change nothing (named in `scripts/engine.env`); they
  stay so the set is the measured one.

### API notes

TensorFold's Zig server: `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/messages` (Anthropic),
`/v1/models`, `/health`, `/metrics` and more. Tool calls work. A conversation's next turn resumes from the kept state of
its earlier turns instead of reading the whole prompt again.

## What was checked

Without a Spark (the details are in this repository's history and can be re-run):

- `patches/` applies with `git apply` onto a fresh clone of TensorFold at `089a18dff` (`scripts/prepare.sh
  --check-patches`), and the result is the deployment's engine tree, which carries the five-expert switch
  (`patches/0033`, off by default), except reworded comments and strings.
- The `docker run` commands `DRY_RUN=1 ./start.sh` prints match the deployment's launcher for both ranks: the same
  docker options, the same 113 container variables (all but the link's netdev, RoCE devices, GID index and the experts
  folder's path, which depend on the machine), and the same `serve` arguments except `--host` (the deployment binds
  127.0.0.1 behind a gateway).
- The image recipe is the deployment's Dockerfile with the source taken from a clone plus `patches/` instead of a
  source tarball of the same tree.
- `tools/exl3_experts_pack.py` lays out the pinned shards exactly as the deployment's folder: from the shards' real
  headers it writes a `model.safetensors.index.json` byte-identical to the deployment's, and the same ten files with
  the same sizes and tensor counts. One output file was fetched by range from the pinned branch and its sha256 equals
  the deployment's. The cut from whole shards (`EXL3_SOURCE=download`) was checked byte for byte on synthetic shards.

Not checked without a Spark: the image build itself, the link detection on real CX7 ports, the copy to the worker, and a
full start. The deployment's engine is built and serving from the same code; this recipe's own image build has not
been run.

## Licenses

This repository's own files (scripts, tools, documentation) are under the Apache License 2.0
([`LICENSE`](LICENSE)); [`NOTICE`](NOTICE) carries the third-party notices. The patches modify TensorFold, which is under
the Apache License 2.0 from v0.6.0 with an MIT notice for code written before it; the TensorFold code they modify or
quote as context stays under TensorFold's licenses. The upstream TensorFold commits in the series stay their authors'
contributions under TensorFold's license. Patches 0002-0003 port changes from MiaAI-Lab's single-Spark recipe under its
MIT license (notice in [`NOTICE`](NOTICE)).

The model files are downloaded from Hugging Face and are not part of this repository:

- **The checkpoint** [`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local)
  is governed by the [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/),
  with the Qwen Community License 1.0 for the base model (its model card).
- **The EXL3 quant** [`turboderp/Qwen3.8-Flash-Next-exl3`](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3) is
  under the Qwen Community License 1.0 (its model card and `LICENSE`).
- **The base model** [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) is under the
  [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE). It is permissive
  (use, modify, distribute, sell, host and fine-tune, keeping its notice), with two commercial conditions:
  1. A commercial product or service using the model (or a derivative) with more than 100,000,000 monthly active users
     or more than US$20,000,000 monthly revenue must display the model's name prominently in its user interface.
  2. A licensee (or affiliate) that runs a Model-as-a-Service or AI Work Assistant business (an independent product
     primarily for AI-assisted coding or office productivity) needs a separate license from Qwen before using the model
     for any commercial purpose; internal use that exposes neither the model, its outputs nor its capabilities to any
     third party is exempt.

  Read the license itself; this summary is not legal advice.

**Third-party software in the image.** The image is built on NVIDIA's PyTorch container
`nvcr.io/nvidia/pytorch:26.07-py3`; the NVIDIA software in it is governed by the NVIDIA Software License Agreement and
the Product-Specific Terms for NVIDIA AI Products, which the container prints at every start. It also contains
libjpeg-turbo (BSD-style and IJG licenses), libwebp (BSD-3-Clause) and TensorFold v1.1.4's grammar library with the
xgrammar and dlpack license files (Apache-2.0). This repository's Apache License covers its own work only.

## Credits

Built on [TensorFold](https://github.com/ashhart/TensorFold) by Ash Hart ([ashhart](https://github.com/ashhart)) and
the TensorFold contributors. The series carries upstream work by laudney (PR #1784, the two-rank head split every
later patch builds on), Armin (PR #1942's DeltaNet lanes and three ported patches), shuuul, Jürgen Schmied,
devopsfortunato and Ash Hart. The scripts are adapted from MiaAI-Lab's two-Spark recipe; patches 0002-0003 port
MiaAI-Lab's single-Spark recipe patches 0001-0017; the GPU top-k candidate idea comes from MiaAI-Lab's DeepSeek-V4.1
recipe (patch 0021); `TF_FLASHNEXT_LATE_SKIP` from Picom-code's pico-dgx-qwen38 patch 0111; the five-expert switch
from azampatti's 5-expert INT4-AutoRound checkpoint. Benchmarks used MiaAI-Lab's sparkDash. The EXL3 format, exllamav3
and the quant are turboderp's; the FP4 checkpoint is NVIDIA's; the model is the Qwen team's. The full list, with
licenses and links, is in [`CREDITS.md`](CREDITS.md).

## Repository layout

```
start.sh, stop.sh   start and stop both ranks
scripts/            config.sh (settings), engine.env (the engine's variables), prepare.sh (image and weights on both
                    Sparks), nodes.sh (ssh and the link), local.sh.example, checkpoint.sha256, exl3-pack.sha256
tools/              exl3_experts_pack.py (the per-rank EXL3 experts folder)
bench/              the prompt-reuse tests and LCQ v2 (bench/README.md)
patches/            0001-0034, applied to TensorFold at TF_REF in filename order
CHANGELOG.md, CREDITS.md, LICENSE, NOTICE
```
