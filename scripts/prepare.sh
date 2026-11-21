#!/usr/bin/env bash
# Prepare both Sparks to serve Qwen3.8-Flash-Next with TensorFold at --tp 2 (rank 0 here, rank 1 on WORKER):
#   1. preflight on both: docker and the GPU, rsync, key-based ssh to the worker, the RoCE link, disk
#   2. the image here: TensorFold at TF_REF with patches/*.patch applied (git apply), built with Zig in NVIDIA's
#      PyTorch container, with libjpeg-turbo, libwebp and TensorFold v1.1.4's grammar library (scripts/config.sh)
#   3. the same image on the worker (docker save | ssh docker load), checked identical
#   4. the checkpoint here (MODEL_ID at MODEL_REVISION, ~96 GiB), checked against scripts/checkpoint.sha256
#   5. the EXL3 experts here (tools/exl3_experts_pack.py, from EXL3_REPO at EXL3_REVISION), checked against
#      scripts/exl3-pack.sha256
#   6. the checkpoint and rank 1's experts on the worker, copied over the link (rsync) and checked
# ./start.sh runs this by itself when needed. Safe to re-run: every step skips work already done.
# Adapted from MiaAI-Lab's Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold scripts/prepare.sh (Apache-2.0; CREDITS.md), with
# the deployment's Dockerfile and this recipe's weights.
# Options:
#   --rebuild               rebuild the image from scratch (docker build --no-cache)
#   --image-only            steps 1-3 only
#   --weights-only          steps 1 and 4-6 only
#   --check-patches [DIR]   no docker, no worker: clone TF_REPO at TF_REF into DIR (default: a new temporary folder)
#                           and apply patches/*.patch as the image build does; prints the result and exits
#   -h, --help              this text
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./scripts/config.sh
source ./scripts/nodes.sh

REBUILD=0; DO_IMAGE=1; DO_WEIGHTS=1; CHECK_PATCHES=""
while (( $# )); do
  case "$1" in
    --rebuild) REBUILD=1 ;;
    --image-only) DO_WEIGHTS=0 ;;
    --weights-only) DO_IMAGE=0 ;;
    --check-patches) CHECK_PATCHES=new; if [[ $# -ge 2 && "$2" != -* ]]; then CHECK_PATCHES=$2; shift; fi ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) die "unknown argument: $1 (see --help)" ;;
  esac
  shift
done
(( DO_IMAGE || DO_WEIGHTS )) || die "--image-only and --weights-only together leave nothing to do"

# apply_patches <dir>: TensorFold at TF_REF in <dir>, every patches/*.patch applied in filename order (the image build's
# own steps, outside docker)
apply_patches() {
  local dir=$1 p n=0
  git init -q "$dir"
  git -C "$dir" remote add origin "$TF_REPO"
  git -C "$dir" fetch -q --depth 1 origin "$TF_REF"
  git -C "$dir" checkout -q --detach FETCH_HEAD
  [[ "$(git -C "$dir" rev-parse HEAD)" == "$TF_REF" ]] || die "$TF_REPO gave $(git -C "$dir" rev-parse HEAD), not $TF_REF"
  for p in patches/*.patch; do
    git -C "$dir" apply --check "$PWD/$p" || die "$p does not apply onto $TF_REF"
    git -C "$dir" apply "$PWD/$p"
    n=$((n + 1))
  done
  log "Applied $n patches onto TensorFold ${TF_REF:0:12} in $dir: $(git -C "$dir" status --porcelain | wc -l) paths changed"
}
if [[ -n "$CHECK_PATCHES" ]]; then
  if [[ "$CHECK_PATCHES" == new ]]; then CHECK_PATCHES=$(mktemp -d "${TMPDIR:-/tmp}/tf-check.XXXXXX")
  else [[ ! -e "$CHECK_PATCHES" ]] || die "$CHECK_PATCHES exists: give a new folder"; fi
  apply_patches "$CHECK_PATCHES"
  exit 0
fi

# ---------------------------------------------------------------- 1. preflight
mkdir -p "$STATE_DIR" "$KERNEL_CACHE" "$HF_CACHE/hub"
exec 9>"$STATE_DIR/prepare.lock"
flock -n 9 || die "another prepare.sh is already running; wait for it or stop it (pgrep -af prepare.sh)"
log "Preflight checks on both Sparks"
for c in docker rsync curl python3 sha256sum flock; do command -v "$c" >/dev/null || die "$c is not installed here"; done
docker info >/dev/null 2>&1 || die "cannot talk to the docker daemon (is your user in the docker group?)"
nvidia-smi -L >/dev/null 2>&1 || warn "nvidia-smi failed here: is the NVIDIA driver working?"
check_workers
need_worker 1
worker 1 'docker info >/dev/null 2>&1' || die "the worker ($WORKER) cannot talk to its docker daemon (docker group?)"
worker 1 'command -v rsync >/dev/null && command -v sha256sum >/dev/null' || die "the worker needs rsync and sha256sum"
worker 1 'nvidia-smi -L >/dev/null 2>&1' || warn "nvidia-smi failed on the worker: is the NVIDIA driver working?"
detect_link
log "Link: head $HEAD_ADDR ($HEAD_DEV, $HEAD_HCAS, GID $HEAD_GID) <-> worker $WORKER_ADDR ($WORKER_DEV, $WORKER_HCAS)"
WORKER_HF=$(worker_hf_cache)
worker 1 "mkdir -p '$WORKER_HF/hub' && test -w '$WORKER_HF/hub'" ||
  die "the worker's $WORKER_HF/hub is not writable (left root-owned by a container? fix its ownership there)"
free_gb() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'; }
worker_free_gb() { worker 1 "df -BG --output=avail '$1' | tail -1 | tr -dc '0-9'"; }

# ---------------------------------------------------------------- 2. image (head)
# The Dockerfile is the one the measured deployment's image was built with, except that it clones TensorFold at TF_REF
# and applies patches/ (the deployment built from the same tree as a source tarball). Stages: the image codecs
# (libjpeg-turbo, libwebp), TensorFold v1.1.4's prebuilt grammar library and its licenses, Zig, then the engine:
# `zig build -Doptimize=fast fatbins native` (the CUDA kernels compiled into fatbins for sm_121, no Triton kernel set).
# Only /opt/tensorfold reaches the serving image, with TensorFold's license files.
if (( DO_IMAGE )); then
  if (( REBUILD )) || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    DOCKER_ROOT=$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)
    have=$(free_gb "$DOCKER_ROOT"); (( have >= IMAGE_FREE_GB )) ||
      die "only ${have} GB free under Docker's root ($DOCKER_ROOT); the build needs ~${IMAGE_FREE_GB} GB (IMAGE_FREE_GB)"
    log "Building $IMAGE: TensorFold ${TF_REF:0:12} + $(compgen -G 'patches/*.patch' | wc -l) patches, Zig $ZIG_VERSION, on $BASE_IMAGE (about 15 minutes)"
    ctx=$(mktemp -d "$STATE_DIR/build.XXXXXX")
    trap 'rm -rf -- "$ctx"' EXIT
    mkdir -p "$ctx/patches"; cp -- patches/*.patch "$ctx/patches/"
    nocache=(); (( REBUILD )) && nocache=(--no-cache)
    docker build "${nocache[@]}" --target serve -t "$IMAGE" \
      --build-arg BASE_IMAGE="$BASE_IMAGE" --build-arg ZIG_VERSION="$ZIG_VERSION" --build-arg ZIG_SHA256="$ZIG_SHA256" \
      --build-arg LIBJPEG_TURBO_VERSION="$LIBJPEG_TURBO_VERSION" --build-arg LIBJPEG_TURBO_SHA256="$LIBJPEG_TURBO_SHA256" \
      --build-arg LIBWEBP_VERSION="$LIBWEBP_VERSION" --build-arg LIBWEBP_SHA256="$LIBWEBP_SHA256" \
      --build-arg TF_RELEASE_VERSION="$TF_RELEASE_VERSION" --build-arg TF_RELEASE_AARCH64_SHA256="$TF_RELEASE_AARCH64_SHA256" \
      --build-arg TF_REPO="$TF_REPO" --build-arg TF_REF="$TF_REF" --build-arg IMAGE_HASH="$(image_hash)" \
      -f - "$ctx" <<'DOCKERFILE'
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE_IMAGE} AS codecs
ARG LIBJPEG_TURBO_VERSION
ARG LIBJPEG_TURBO_SHA256
ARG LIBWEBP_VERSION
ARG LIBWEBP_SHA256
RUN mkdir -p /src /opt/codecs && cd /src && \
    curl -fsSL -o jt.tar.gz "https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/${LIBJPEG_TURBO_VERSION}/libjpeg-turbo-${LIBJPEG_TURBO_VERSION}.tar.gz" && \
    echo "${LIBJPEG_TURBO_SHA256}  jt.tar.gz" | sha256sum -c - && \
    tar -xzf jt.tar.gz && cmake -S libjpeg-turbo-${LIBJPEG_TURBO_VERSION} -B jt-build -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_SHARED=ON -DENABLE_STATIC=OFF -DWITH_TURBOJPEG=ON -DCMAKE_INSTALL_PREFIX=/opt/codecs/jpeg -DCMAKE_INSTALL_LIBDIR=lib && \
    cmake --build jt-build -j"$(nproc)" && cmake --install jt-build && \
    test -e /opt/codecs/jpeg/lib/libturbojpeg.so.0
RUN cd /src && \
    curl -fsSL -o webp.tar.gz "https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-${LIBWEBP_VERSION}.tar.gz" && \
    echo "${LIBWEBP_SHA256}  webp.tar.gz" | sha256sum -c - && \
    tar -xzf webp.tar.gz && cd libwebp-${LIBWEBP_VERSION} && \
    ./configure --prefix=/opt/codecs/webp --enable-libwebpdecoder --disable-static --disable-gl --disable-sdl --disable-png --disable-jpeg --disable-tiff --disable-gif --disable-wic && \
    make -j"$(nproc)" && make install && \
    test -e /opt/codecs/webp/lib/libwebpdecoder.so.3

FROM ${BASE_IMAGE} AS grammar
ARG TF_RELEASE_VERSION
ARG TF_RELEASE_AARCH64_SHA256
RUN mkdir -p /opt/tfrel && cd /tmp && \
    curl -fsSL -o tf-rel.tar.gz "https://github.com/ashhart/TensorFold/releases/download/v${TF_RELEASE_VERSION}/tensorfold-${TF_RELEASE_VERSION}-linux-aarch64.tar.gz" && \
    echo "${TF_RELEASE_AARCH64_SHA256}  tf-rel.tar.gz" | sha256sum -c - && \
    tar -xzf tf-rel.tar.gz -C /opt/tfrel --strip-components=1 \
      tensorfold-${TF_RELEASE_VERSION}-linux-aarch64/lib/libtfgrammar.so \
      tensorfold-${TF_RELEASE_VERSION}-linux-aarch64/LICENSES/xgrammar-Apache-2.0.txt \
      tensorfold-${TF_RELEASE_VERSION}-linux-aarch64/LICENSES/xgrammar-NOTICE.txt \
      tensorfold-${TF_RELEASE_VERSION}-linux-aarch64/LICENSES/dlpack-Apache-2.0.txt && \
    test -s /opt/tfrel/lib/libtfgrammar.so

FROM ${BASE_IMAGE} AS zig
ARG ZIG_VERSION
ARG ZIG_SHA256
RUN curl -fsSL -o /tmp/zig.tar.xz "https://ziglang.org/download/${ZIG_VERSION}/zig-aarch64-linux-${ZIG_VERSION}.tar.xz" && \
    echo "${ZIG_SHA256}  /tmp/zig.tar.xz" | sha256sum -c - && \
    mkdir -p /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 && rm /tmp/zig.tar.xz && \
    /opt/zig/zig version

FROM zig AS build
ARG TF_REPO
ARG TF_REF
RUN git init -q /tensorfold && cd /tensorfold && git remote add origin "${TF_REPO}" && \
    git fetch -q --depth 1 origin "${TF_REF}" && git checkout -q --detach FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = "${TF_REF}"
COPY patches /opt/tf-patches
RUN cd /tensorfold && \
    for p in /opt/tf-patches/*.patch; do echo "applying $p"; git apply --check "$p" && git apply "$p" || exit 1; done
RUN cd /tensorfold && /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=fast --prefix /opt/tensorfold \
      -j"$(nproc)" fatbins native && \
    test -x /opt/tensorfold/native/bin/tensorfold-native
RUN mkdir -p /opt/tensorfold/share/doc/tensorfold && cd /tensorfold && \
    for f in LICENSE LICENSE.md LICENSES NOTICE THIRD_PARTY_NOTICES.md; do \
      if [ -e "$f" ]; then cp -a "$f" /opt/tensorfold/share/doc/tensorfold/ || exit 1; fi; done

FROM ${BASE_IMAGE} AS serve
COPY --from=build /opt/tensorfold /opt/tensorfold
COPY --from=codecs /opt/codecs/jpeg/lib/libturbojpeg.so* /opt/tensorfold/lib/
COPY --from=codecs /opt/codecs/webp/lib/libwebpdecoder.so* /opt/tensorfold/lib/
COPY --from=grammar /opt/tfrel/lib/libtfgrammar.so /opt/tensorfold/native/lib/libtfgrammar.so
COPY --from=grammar /opt/tfrel/LICENSES /opt/tensorfold/share/doc/tensorfold/LICENSES-grammar
RUN ln -s /opt/tensorfold/native/bin/tensorfold-native /usr/local/bin/tensorfold-native
ARG IMAGE_HASH
LABEL tf.recipe.image=${IMAGE_HASH} tf.kernels=embedded
ENV HF_HOME=/root/.cache/huggingface CUDA_CACHE_PATH=/cache/nv \
    TENSORFOLD_LIBTURBOJPEG=/opt/tensorfold/lib/libturbojpeg.so.0 TENSORFOLD_LIBWEBP=/opt/tensorfold/lib/libwebpdecoder.so.3
WORKDIR /workspace
DOCKERFILE
    rm -rf -- "$ctx"; trap - EXIT
  else
    log "Image $IMAGE already built (--rebuild builds it again)"
  fi
  docker run --rm --network none --entrypoint test "$IMAGE" -x /opt/tensorfold/native/bin/tensorfold-native ||
    die "$IMAGE has no tensorfold-native"

  # -------------------------------------------------------------- 3. image (worker)
  image_id=$(image_ident "$IMAGE")
  if [[ "$(worker_image_ident 1 "$IMAGE")" != "$image_id" ]]; then
    wroot=$(worker 1 "docker info -f '{{.DockerRootDir}}'" 2>/dev/null || echo /var/lib/docker)
    wfree=$(worker_free_gb "$wroot")
    (( wfree >= IMAGE_FREE_GB )) || die "only ${wfree} GB free under the worker's Docker root ($wroot); the image needs ~${IMAGE_FREE_GB} GB"
    log "Copying $IMAGE to the worker (docker save | docker load)"
    docker save "$IMAGE" | worker 1 docker load >/dev/null
    [[ "$(worker_image_ident 1 "$IMAGE")" == "$image_id" ]] || die "the worker's $IMAGE differs from the head's"
  fi
  log "Image $IMAGE identical on both Sparks"
fi

if (( DO_WEIGHTS )); then
  # -------------------------------------------------------------- 4. checkpoint (head)
  # hf_download <repo> <revision> [include pattern...]: into HF_CACHE/hub with the hf CLI, else this host's
  # huggingface_hub. HF_TOKEN (optional: both repositories are public) goes by name only.
  hf_download() {
    local repo=$1 rev=$2; shift 2
    if command -v hf >/dev/null; then
      local inc=(); for p in "$@"; do inc+=(--include "$p"); done
      hf download "$repo" --revision "$rev" --cache-dir "$HF_CACHE/hub" "${inc[@]}" >/dev/null
    elif python3 -c 'import huggingface_hub' 2>/dev/null; then
      python3 -I -c 'import sys; from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], revision=sys.argv[2], cache_dir=sys.argv[3], allow_patterns=sys.argv[4:] or None)' \
        "$repo" "$rev" "$HF_CACHE/hub" "$@" >/dev/null
    else
      die "neither the hf CLI nor huggingface_hub is installed here (pip install -U huggingface_hub)"
    fi
  }
  snap=$(model_dir)
  if [[ ! -f "$STATE_DIR/checkpoint-$MODEL_REVISION.ok" ]]; then
    if [[ ! -f "$snap/config.json" ]]; then
      have=$(free_gb "$HF_CACHE"); (( have >= CKPT_GIB + 5 )) ||
        warn "only ${have} GB free under $HF_CACHE; the checkpoint is ~$CKPT_GIB GiB (a partial download resumes)"
    fi
    log "Downloading $MODEL_ID @ ${MODEL_REVISION:0:8} into $HF_CACHE/hub (resumes if interrupted)"
    hf_download "$MODEL_ID" "$MODEL_REVISION" || die "$MODEL_ID: the download failed"
    dir="$HF_CACHE/hub/models--${MODEL_ID//\//--}"
    [[ -f "$dir/refs/main" ]] || { mkdir -p "$dir/refs"; printf %s "$MODEL_REVISION" > "$dir/refs/main"; }
    [[ -f "$snap/config.json" ]] || die "no snapshot $snap after the download"
    log "Checking $MODEL_ID against scripts/checkpoint.sha256 (reads ~$CKPT_GIB GiB once)"
    grep -v '^#' scripts/checkpoint.sha256 | (cd "$snap" && sha256sum -c --quiet) || die "the checkpoint at $snap does not match scripts/checkpoint.sha256"
    touch "$STATE_DIR/checkpoint-$MODEL_REVISION.ok"
  fi
  log "Checkpoint OK: $snap"

  # -------------------------------------------------------------- 5. EXL3 experts (head)
  pack=$(pack_dir)
  PACK_PINS=$PWD/scripts/exl3-pack.sha256
  if [[ "$EXL3_EXPERTS" == 1 && ! -f "$pack/.ok-head" ]]; then
    mkdir -p "$pack"
    case "$EXL3_SOURCE" in
      range)
        log "Fetching the routed experts of $EXL3_REPO @ ${EXL3_REVISION:0:8} ($EXL3_BRANCH) by range into $pack (~37.5 GiB, resumes)"
        python3 -I tools/exl3_experts_pack.py fetch --repo "$EXL3_REPO" --rev "$EXL3_REVISION" --shards "$EXL3_SHARDS" \
          --out "$pack" --ranks 0,1 --pins "$PACK_PINS" --limit-rate "$EXL3_LIMIT_RATE" ;;
      download)
        log "Downloading $EXL3_REPO @ ${EXL3_REVISION:0:8} ($EXL3_BRANCH), its model shards (~43.4 GiB, resumes)"
        hf_download "$EXL3_REPO" "$EXL3_REVISION" 'model-*.safetensors' || die "$EXL3_REPO: the download failed"
        src="$HF_CACHE/hub/models--${EXL3_REPO//\//--}/snapshots/$EXL3_REVISION"
        python3 -I tools/exl3_experts_pack.py extract --repo "$EXL3_REPO" --rev "$EXL3_REVISION" --shards "$EXL3_SHARDS" \
          --src "$src" --out "$pack" --ranks 0,1 --pins "$PACK_PINS" --check-shards ;;
      *) die "EXL3_SOURCE is range or download, not $EXL3_SOURCE" ;;
    esac
    touch "$pack/.ok-head"
  fi
  [[ "$EXL3_EXPERTS" == 1 ]] && log "EXL3 experts OK: $pack"

  # -------------------------------------------------------------- 6. the worker's copy
  # The checkpoint's blobs and its snapshot's links (the cache layout huggingface_hub keeps), then rank 1's experts.
  wdir="$WORKER_HF/hub/models--${MODEL_ID//\//--}"
  hdir="$HF_CACHE/hub/models--${MODEL_ID//\//--}"
  manifest=$(cd "$snap" && find -L . -type f -printf '%P %s\n' | LC_ALL=C sort)
  have=$(worker 1 "cd '$wdir/snapshots/$MODEL_REVISION' 2>/dev/null && find -L . -type f -printf '%P %s\n' | LC_ALL=C sort" || true)
  if [[ "$have" != "$manifest" ]]; then
    wfree=$(worker_free_gb "$WORKER_HF")
    (( wfree >= CKPT_GIB + 5 )) || warn "only ${wfree} GB free under $WORKER_HF on the worker; the checkpoint is ~$CKPT_GIB GiB"
    log "Copying $MODEL_ID @ ${MODEL_REVISION:0:8} to the worker over the link (resumes)"
    lists=$(mktemp -d "$STATE_DIR/lists.XXXXXX")
    (cd "$hdir" && find "snapshots/$MODEL_REVISION" -type l -printf '%l\n' | sed 's#^\(\.\./\)*##' | LC_ALL=C sort -u) > "$lists/blobs"
    worker 1 "mkdir -p '$wdir/refs' '$wdir/snapshots'"
    read -r -a ropts <<<"${RSYNC_OPTS:-}"
    rsync -a -L --partial --files-from="$lists/blobs" "$hdir/" "$WORKER:$wdir/" -e "ssh -o BatchMode=yes" "${ropts[@]}"
    rsync -a "$hdir/snapshots/$MODEL_REVISION" "$WORKER:$wdir/snapshots/" -e "ssh -o BatchMode=yes"
    worker 1 "test -f '$wdir/refs/main' || printf %s '$MODEL_REVISION' > '$wdir/refs/main'"
    rm -rf -- "$lists"
    have=$(worker 1 "cd '$wdir/snapshots/$MODEL_REVISION' && find -L . -type f -printf '%P %s\n' | LC_ALL=C sort")
    [[ "$have" == "$manifest" ]] || die "the worker's copy of $MODEL_ID differs from the head's after the copy"
    worker 1 "rm -f '$wdir/.recipe-checked-$MODEL_REVISION'"
  fi
  if ! worker 1 "test -f '$wdir/.recipe-checked-$MODEL_REVISION'"; then
    log "Checking the worker's checkpoint against scripts/checkpoint.sha256"
    grep -v '^#' scripts/checkpoint.sha256 | worker 1 "cd '$wdir/snapshots/$MODEL_REVISION' && sha256sum -c --quiet" ||
      die "the worker's checkpoint does not match scripts/checkpoint.sha256"
    worker 1 "touch '$wdir/.recipe-checked-$MODEL_REVISION'"
  fi
  log "Worker has $MODEL_ID @ ${MODEL_REVISION:0:8}"
  if [[ "$EXL3_EXPERTS" == 1 ]]; then
    wpack="$WORKER_HF/$EXL3_PACK"
    r1=$(grep -o 'experts-r1-[0-9]*-of-[0-9]*\.safetensors' "$pack/SHA256SUMS" | LC_ALL=C sort -u)
    [[ -n "$r1" ]] || die "$pack/SHA256SUMS names no rank 1 files"
    if ! worker 1 "test -f '$wpack/.ok-worker'"; then
      log "Copying rank 1's experts to the worker ($wpack)"
      worker 1 "mkdir -p '$wpack'"
      for f in model.safetensors.index.json manifest.json SHA256SUMS $r1; do
        [[ -f "$pack/$f" ]] || die "$pack/$f is missing here (EXL3_PRUNE removed it? delete $pack/.ok-head and rerun)"
        rsync -a --partial "$pack/$f" "$WORKER:$wpack/$f" -e "ssh -o BatchMode=yes"
      done
      grep -E ' experts-r1-' "$pack/SHA256SUMS" | worker 1 "cd '$wpack' && sha256sum -c --quiet" ||
        die "rank 1's experts on the worker do not match SHA256SUMS"
      worker 1 "touch '$wpack/.ok-worker'"
    fi
    log "Worker has rank 1's EXL3 experts"
    # rank 0 reads only its own files: EXL3_PRUNE=1 (the default) removes rank 1's from the head once the worker has them
    if [[ "${EXL3_PRUNE:-1}" == 1 ]]; then
      for f in $r1; do rm -f -- "${pack:?}/$f"; done
    fi
  fi
fi

(( DO_IMAGE && DO_WEIGHTS )) && prepared_state > "$PREPARED_MARKER"
log "Done: both Sparks are ready. Start the server with ./start.sh (port $PORT)."
