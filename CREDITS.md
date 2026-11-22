# Credits

This repository is a patch series on TensorFold plus the scripts that build and serve it. Most of what makes it work
was built by others. Its own work is licensed under the Apache License 2.0 ([`LICENSE`](LICENSE)); [`NOTICE`](NOTICE)
carries the third-party notices that go with it.

## Inference engine

- **[TensorFold](https://github.com/ashhart/TensorFold)** and its Zig engine `tensorfold-native`, by Ash Hart
  ([ashhart](https://github.com/ashhart)) and the
  [TensorFold contributors](https://github.com/ashhart/TensorFold/graphs/contributors) (Apache-2.0 from v0.6.0; code
  written before v0.6.0 keeps its MIT notice): the engine, its CUDA runtime and kernels, the Flash Next family, MTP
  drafting, the prompt cache and the OpenAI-compatible server. `scripts/prepare.sh` builds it from upstream `main` at
  `089a18dff289d4c58043c1957ad0d7fe73d814b3` with `patches/` applied. TensorFold credits the projects it adapts in its
  [third-party notices](https://github.com/ashhart/TensorFold/blob/main/THIRD_PARTY_NOTICES.md).
- **Upstream TensorFold work carried in the series**, each patch with its author in `From:` and its upstream source in
  the message:
  - **laudney** ([laudney](https://github.com/laudney)): Flash Next at `--tp 2`, TensorFold
    [PR #1784](https://github.com/ashhart/TensorFold/pull/1784) (`patches/0001`, the two-rank head split every later
    patch builds on, as the PR's head of 2026-11-02), and the ports `0014` (a stream that finishes on its first token)
    and `0021` (exact image identities in prompt states).
  - **Armin** ([Arminova](https://github.com/Arminova)): the shared pass's DeltaNet lanes of
    [PR #1942](https://github.com/ashhart/TensorFold/pull/1942), adapted to a rank's half heads in `0003`; the ports
    `0018` (#2244, the grouped draft head's page table), `0023` and `0024` (parts of #1764: the wide router tile, the
    sparse-index select).
  - **shuuul** ([shuuul](https://github.com/shuuul)): staged synchronous downloads on integrated GPUs, `0008`-`0011`.
  - **Ash Hart** ([ashhart](https://github.com/ashhart)): `0012`, `0013`, `0015`-`0017`, `0019`, `0020`, `0026` (staged
    readbacks, prompt-cache restore retries, cached-token counts, trailing tool calls, newest-state eviction).
  - **Jürgen Schmied** ([jschmied](https://github.com/jschmied)): `0022` (the MTP head's experts on the call's last row).
  - **devopsfortunato** ([devopsfortunato](https://github.com/devopsfortunato)): `0028` (an unstarted resumed turn
    ranked by its uncached tail).
  - The idea of upstream PR #1792 (the shared partial summed where the down projection wrote it), applied in `0003`.
- **MiaAI-Lab's single-Spark recipe**,
  [Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/tree/20e15e41f2f514a0d287710f4665ee1b5009ba41)
  (MIT, Copyright (c) 2026 MiaAI-Lab): its patches 0001-0017 are ported in `patches/0002` and `0003`, each change
  naming its recipe patch (vision tower, image budget, the FP8 KV cache, FP8 reader prefetch, prefill speed, sampled
  GPU top-k candidates, the shared n-gram gather, host draws on worker threads, prompt HC fusion, the MX128 prompt tile,
  the draft stop rule, the prefill tail and draft read-ahead). Its MIT notice is in [`NOTICE`](NOTICE).
- **MiaAI-Lab's DeepSeek-V4.1 recipe**,
  [DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold):
  its patch 0021 (each row's best candidates kept on the GPU so the host draws among k scores) is the idea behind the
  candidate kernel in `patches/0002`, written anew for this engine.
- **Picom-code's** [pico-dgx-qwen38](https://github.com/Picom-code/pico-dgx-qwen38) (Apache-2.0): `TF_FLASHNEXT_LATE_SKIP`
  in `patches/0031` is ported from the idea of its patch 0111 (q38-drafted-windows-skip-deltanet-end-state-writes),
  rewritten for this tree; no code copied.
- **azampatti** ([azampatti](https://huggingface.co/azampatti)): the five-expert switch (`patches/0033`,
  `TF_FLASHNEXT_EXPERTS_TOPK`, off by default) follows the idea of the 5-expert checkpoint
  [`azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound);
  no weights or code of it are used.
- **turboderp** ([turboderp](https://huggingface.co/turboderp)): the EXL3 format and
  [ExLlamaV3](https://github.com/turboderp-org/exllamav3) (MIT, Copyright (c) 2025 Turboderp), which TensorFold's EXL3
  decoders and kernels implement separately (TensorFold's third-party notices), and the quant this recipe serves the
  routed experts from, [`turboderp/Qwen3.8-Flash-Next-exl3`](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3)
  (branch `2.68bpw_h6_ng8`).
- **[Zig](https://ziglang.org)** by the Zig Software Foundation and the Zig contributors (MIT): the language and compiler
  TensorFold's engine is built with (0.17.0).

## Model

- **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** by the [Qwen team](https://huggingface.co/Qwen)
  (Alibaba): the model, its chat template and its MTP head, under the
  [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE).
- **[NVIDIA](https://huggingface.co/nvidia)**: the FP4 checkpoint this recipe serves,
  [`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local), governed by the
  [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/).
- **turboderp**: the EXL3 quant of the routed experts (above), under the Qwen Community License 1.0.

The weights are not part of this repository; `scripts/prepare.sh` downloads them from Hugging Face at pinned revisions.

## Recipe

- **MiaAI-Lab's two-Spark recipe**,
  [Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold)
  (Apache-2.0, Copyright 2026 MiaAI-Lab): this repository's layout and README follow it, and these files are adapted
  from its own:
  - `scripts/config.sh` (the settings order: environment, `scripts/local.sh`, `.env`; logging helpers) from its
    `scripts/config.sh`;
  - `scripts/nodes.sh` (ssh to the worker, saved logs, the link and RoCE rail detection, the NCCL environment) from its
    `scripts/nodes.sh`, without the NFS path;
  - `start.sh` and `stop.sh` from its own (the start steps, the stop and log saving), with this recipe's launch;
  - `scripts/prepare.sh` from its `scripts/prepare.sh` (the preflight, the patch apply in the image build, the image
    copy and the checkpoint copy to the worker), with this recipe's Dockerfile and weights;
  - `scripts/local.sh.example` from its own.
  The changes are described in each file's header and in [`CHANGELOG.md`](CHANGELOG.md).
- **[sparkDash](https://github.com/MiaAI-Lab/sparkDash)** by MiaAI-Lab (Apache-2.0): the benchmark the README's decode and
  prompt-reading figures were measured with. `patches/0002` also reports the engine's running totals on `/health` under
  the names sparkDash reads. No sparkDash code is in this repository.
- The Dockerfile in `scripts/prepare.sh` and the launch in `start.sh` are those of the deployment the README's numbers
  come from.

## Runtime stack

- **[NVIDIA PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)**
  (`nvcr.io/nvidia/pytorch:26.07-py3`), the image's base, with CUDA (nvcc builds the kernels), NCCL and the CUDA
  forward-compatibility library. Governed by the NVIDIA Software License Agreement and the Product-Specific Terms for
  NVIDIA AI Products.
- **[libjpeg-turbo](https://libjpeg-turbo.org)** (IJG and BSD-3-Clause licenses) and
  **[libwebp](https://chromium.googlesource.com/webm/libwebp)** (BSD-3-Clause): image decoding.
- **TensorFold v1.1.4's grammar library** (`libtfgrammar.so`), with the **[xgrammar](https://github.com/mlc-ai/xgrammar)**
  and **[dlpack](https://github.com/dmlc/dlpack)** license files (Apache-2.0) it ships with.
- **[NCCL](https://github.com/NVIDIA/nccl)** (BSD-3-Clause) and **[rdma-core](https://github.com/linux-rdma/rdma-core)**:
  the two ranks' exchanges over the ConnectX-7 RoCE link.
- **[Hugging Face Hub](https://huggingface.co/)**: model hosting, the `hf` CLI and `huggingface_hub` (Apache-2.0).
- **[Docker](https://www.docker.com/)** and the
  **[NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)** (Apache-2.0).

## Benchmark text

- **[Project Gutenberg](https://www.gutenberg.org)** and its volunteers: the public-domain books (in the US) whose text
  is the filler of the reuse tests in `bench/` (`bench/corpus/books.tsv` lists each one; the Project Gutenberg licence
  and trademark text is removed, see `bench/CORPUS-LICENSE`).

## Hardware

- **[NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)** (GB10 Grace Blackwell, 128 GB
  unified memory), two of them linked by their ConnectX-7 ports.

If something here is missing or credited wrongly, please open an issue.
