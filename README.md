# Qwen3.8-Flash-Next on two DGX Sparks with TensorFold, EXL3 experts

Serve **Qwen3.8-Flash-Next** from two NVIDIA DGX Sparks (GB10, 128 GB each, linked by their ConnectX-7 ports) through
an OpenAI-compatible API, with [TensorFold](https://github.com/ashhart/TensorFold)'s Zig engine (`tensorfold-native`)
split across the two Sparks at `--tp 2`: one rank on each.

This is the serving setup of a deployment that serves coding agents around the clock: **24 streams at once, each with
a 409,600-token window** (YaRN factor 2 over the trained 262,144), an **NVFP4 KV cache**, a **45 GiB prompt cache** that
keeps conversation states between agent turns, **routed experts in EXL3** (turboderp's 2.68 bpw quant) on NVIDIA's FP4
checkpoint, MTP drafts, image input and tool calls. The engine changes it needs are a series of 33 patches on upstream
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
The scripts in this form have not yet been run end to end on a pair of Sparks: treat the first run as a test, and
report what breaks.

## Performance and accuracy

All figures below were measured on one pair of DGX Sparks in controlled runs (**lab**) or read from the deployment's
own logs under real agent traffic (**real**). **Every Spark ran with its GPU clock capped at 1,800 MHz**
(`nvidia-smi -lgc 300,1800`; about 1.79 GHz under load), chosen for boxes that run around the clock; `nvidia-smi`
reports a default of 2,418 MHz. This recipe does not set the cap, so an uncapped pair may differ, most likely in prompt
reading, which is bound by compute. Every before and after below ran on the same Sparks at the same cap.

### Decode (lab)

The benchmark ([sparkDash](https://github.com/MiaAI-Lab/sparkDash)) sends a short prose prompt (~80 tokens)
and asks for 400 tokens at temperature 0.7 with thinking on, so this is close to decode-only load, not agent traffic.
Whole-pair decode, tokens a second, 16 November:

| streams | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| **this recipe** (EXL3 experts, 24 × 409,600) | 78 | 101 | 171 | 260 | **379** |

- This row is the engine as it served on 16 November; changes rolled after that day (among them beside packing and
  reply-keep) are in the series but not in this row. The series' own engine on 20 November made 75 / 105 / 171 / 258 /
  375 on the same benchmark ([Five experts](#five-experts-off-by-default)).
- The same engine with the checkpoint's own NVFP4 experts in place of EXL3 (`EXL3_EXPERTS=0`) made 60 / 94 / 162 /
  232 / 335 the same day.

### Prompt reading (lab) — being re-measured

**Being re-measured:** a new run of this table is in progress, and these figures from 16 November may change. One
request at a time, tokens a second:

| | 4K | 32K | 131K | 262K |
|---|---:|---:|---:|---:|
| this recipe | 4,147 | 3,830 | 3,714 | 3,617 |

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
- On both: Docker with the NVIDIA container runtime (your user in the `docker` group), `rsync`, `sha256sum`. On the
  head also `curl`, `python3`, and the `hf` CLI or Python's `huggingface_hub`.
- **Disk.** On each Spark ~96 GiB for the checkpoint under `~/.cache/huggingface`, the EXL3 experts of its rank
  (~19 GiB on the head, ~18 GiB on the worker; the head holds both ranks', ~37.5 GiB, until the worker has its copy),
  and the image under Docker's root (`IMAGE_FREE_GB`, 40 GB asked for; the build's own size is not measured here).
  `EXL3_SOURCE=download` also keeps the six EXL3 shards (~43.4 GiB) on the head.
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
   of TensorFold's v1.1.4 release (sha256-checked), with their license files. The CUDA kernels are compiled into the
   build (no separate kernel set).
3. The same image on the worker (`docker save | docker load`), checked identical.
4. The checkpoint at its pin, into `~/.cache/huggingface`, checked file by file against `scripts/checkpoint.sha256`
   (the Hub's sha256 of every large file).
5. The EXL3 experts into `~/.cache/huggingface/exl3-experts-2.68bpw` by `tools/exl3_experts_pack.py` (below), each
   output file checked against `scripts/exl3-pack.sha256`.
6. The worker's copy: the checkpoint over the link with `rsync` (then checked against `scripts/checkpoint.sha256`
   there), and rank 1's expert files with the index (checked against `SHA256SUMS` there). Rank 1's files are then
   removed from the head (`EXL3_PRUNE=0` keeps them).

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
- `EXL3_SOURCE=download`: downloads the six shards (~43.4 GiB), checks each against the Hub's sha256, and cuts them.

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
value; `ENGINE_UNSET="A B"` leaves names out. `TENSORFOLD_API_KEY` (the server's API keys) goes to rank 0 by name only.

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
  --check-patches`), and the result is the deployment's engine tree plus the five-expert switch (`patches/0033`, off
  by default), except reworded comments and strings.
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
later patch builds on), Armin (PR #1942's DeltaNet lanes and three ported commits), shuuul, Jürgen Schmied,
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
patches/            0001-0033, applied to TensorFold at TF_REF in filename order
CHANGELOG.md, CREDITS.md, LICENSE, NOTICE
```
