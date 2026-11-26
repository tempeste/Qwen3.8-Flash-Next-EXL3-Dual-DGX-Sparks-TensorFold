# Changelog

Every change to this recipe, newest first.

## 1.1 (26 Nov 2026): mixed agent load, current single-Spark baseline

- **Mixed agent load** (README, [Mixed agent load](README.md#mixed-agent-load-lab)): 16 conversations of 100K tokens
  or more at once, 1,024-token replies, against the single-Spark recipe at its published settings and at
  8 × 163,840 a copy; 24 conversations up to 262,144 and up to 409,600 on this recipe alone.
- **`bench/agentload.py`**: the seeded agent workload behind those tables, with the plans and commands to rerun them
  (`bench/README.md`).
- **Baseline**: the table at the top compares with the single-Spark recipe's current version (TensorFold v1.2.7,
  measured 24 Nov). The gains from the starting point still start from the version this work began with (TensorFold
  v1.1.4, 22 Nov); both versions are in [Where this work started](README.md#where-this-work-started).

## 1.0 (23 Nov 2026): first version

- **Experts already in place**: `scripts/prepare.sh` uses an `EXL3_PACK` folder that already holds `manifest.json`
  and, with the sums in `scripts/exl3-pack.sha256`, the index and a Spark's own rank's files (on the head also rank 1's
  while the worker lacks them), instead of fetching or copying them again. The worker's copy is checked against those
  pins, not the head's `SHA256SUMS`, which is optional. `.ok-head` and `.ok-worker` are dropped when a pinned file is
  missing or of another size; prepare.sh stops when the pins file lacks a file or index line; the fetch refuses to
  write through a symlink or hard link, and re-fetches the ranges of an output file it had to create again (rank 1's
  after `EXL3_PRUNE`) even when `state/done.txt` lists them.
- **The engine** (`patches/0001`-`0034`): TensorFold upstream `main` at `089a18dff` plus Flash Next at `--tp 2`
  (upstream PR #1784), two-rank KV formats and prompt-cache snapshots, collectives and shared decode rounds, routed
  experts from EXL3, prompt passes and mixed rounds, the prompt pipeline, batched picks and shared kept rows, 19 ported
  upstream commits (besides PR #1784), relaxed DeltaNet paths and prompt pieces beside decode, picture prompts at
  text speed, small decode tiles and beside packing, reply-keep, the five-expert switch (off by default), and the main stream one priority
  step above the least with beside on, below the split-sum stream. Each patch's message lists the flags it adds and
  their defaults.
- **The serving setup** of the deployment the README's numbers come from: 24 streams × 409,600 tokens, NVFP4 KV cache,
  45 GiB prompt cache, EXL3 routed experts on NVIDIA's FP4 checkpoint, MTP drafts, image input, and its 105 engine
  variables in `scripts/engine.env`.
- **The image**: the deployment's Dockerfile (Zig 0.17.0 on `nvcr.io/nvidia/pytorch:26.07-py3`, libjpeg-turbo
  3.1.4.1, libwebp 1.6.0, TensorFold v1.1.4's grammar library), from a clone of TensorFold plus `patches/`.
- **The weights**: `nvidia/Qwen3.8-Flash-Next-FP4-Local` at `b9ce18a6`, checked against the Hub's sha256; the routed
  experts of `turboderp/Qwen3.8-Flash-Next-exl3` (`2.68bpw_h6_ng8` at `d6601133`), fetched by range or cut from the
  shards by `tools/exl3_experts_pack.py` and checked against the deployment's own sums.
- **The prompt-reuse tests** (`bench/`): warm reuse, shared opening and cache retention (`reuseprobe.py`), and the
  long-context accuracy test LCQ v2 (`lcqgen.py`, `lcq.py`, `lcqstats.py`), for any OpenAI-compatible server; their
  filler is cut from public-domain Project Gutenberg books that `fetch_corpus.py` rebuilds from pinned files.
- **The scripts**, adapted from MiaAI-Lab's Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold: `start.sh`, `stop.sh`,
  `scripts/config.sh`, `scripts/nodes.sh`, `scripts/prepare.sh`. Changed from it: this recipe's launch (the
  deployment's docker options and serve arguments, the engine variables from `scripts/engine.env`, the CUDA
  forward-compatibility library, a checkpoint identity instead of hashing the weights at start), its image and weights,
  the EXL3 experts per rank, a dry run that works without the link; no NFS weights, no prebuilt image, no arguments
  check in a throwaway container, no foreground mode.
