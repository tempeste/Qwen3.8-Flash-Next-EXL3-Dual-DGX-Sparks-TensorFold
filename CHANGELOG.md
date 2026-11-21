# Changelog

Every change to this recipe, newest first.

## Unreleased: first version

- **The engine** (`patches/0001`-`0033`): TensorFold upstream `main` at `089a18dff` plus Flash Next at `--tp 2`
  (upstream PR #1784), two-rank KV formats and prompt-cache snapshots, collectives and shared decode rounds, routed
  experts from EXL3, prompt passes and mixed rounds, the prompt pipeline, batched picks and shared kept rows, 19 ported
  upstream commits (besides PR #1784), relaxed DeltaNet paths and prompt pieces beside decode, picture prompts at text speed, small decode
  tiles and beside packing, reply-keep, and the five-expert switch (off by default). Each patch's message lists the
  flags it adds and their defaults.
- **The serving setup** of the deployment the README's numbers come from: 24 streams × 409,600 tokens, NVFP4 KV cache,
  45 GiB prompt cache, EXL3 routed experts on NVIDIA's FP4 checkpoint, MTP drafts, image input, and its 105 engine
  variables in `scripts/engine.env`.
- **The image**: the deployment's Dockerfile (Zig 0.17.0 on `nvcr.io/nvidia/pytorch:26.07-py3`, libjpeg-turbo
  3.1.4.1, libwebp 1.6.0, TensorFold v1.1.4's grammar library), from a clone of TensorFold plus `patches/`.
- **The weights**: `nvidia/Qwen3.8-Flash-Next-FP4-Local` at `b9ce18a6`, checked against the Hub's sha256; the routed
  experts of `turboderp/Qwen3.8-Flash-Next-exl3` (`2.68bpw_h6_ng8` at `d6601133`), fetched by range or cut from the
  shards by `tools/exl3_experts_pack.py` and checked against the deployment's own sums.
- **The scripts**, adapted from MiaAI-Lab's Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold: `start.sh`, `stop.sh`,
  `scripts/config.sh`, `scripts/nodes.sh`, `scripts/prepare.sh`. Changed from it: this recipe's launch (the
  deployment's docker options and serve arguments, the engine variables from `scripts/engine.env`, the CUDA
  forward-compatibility library, a checkpoint identity instead of hashing the weights at start), its image and weights,
  the EXL3 experts per rank, a dry run that works without the link; no NFS weights, no prebuilt image, no arguments
  check in a throwaway container, no foreground mode.
