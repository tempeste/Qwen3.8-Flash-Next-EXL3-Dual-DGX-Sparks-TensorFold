# Shared settings for start.sh, stop.sh and scripts/*.sh. Adapted from MiaAI-Lab's
# Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold (Apache-2.0; CREDITS.md). A setting's value comes from the first of these
# that sets it:
#   1. the environment: `PORT=9000 ./start.sh`
#   2. scripts/local.sh (this setup's own values, above all WORKER; sourced as bash), then ./.env (KEY=value lines,
#      read, never run): both are yours, not the repository's; where both set a key, local.sh wins
#   3. the defaults below, and for the engine's own variables scripts/engine.env
# shellcheck shell=bash disable=SC2034
_cfg_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -f "$_cfg_root/scripts/local.sh" ]]; then
  declare -A _cfg_env=()
  while IFS= read -r _n; do _cfg_env[$_n]=${!_n}; done < <(compgen -e)
  # shellcheck source=/dev/null
  source "$_cfg_root/scripts/local.sh"
  for _n in "${!_cfg_env[@]}"; do [[ "${!_n-}" == "${_cfg_env[$_n]}" ]] || export "$_n=${_cfg_env[$_n]}"; done
  unset _cfg_env
fi
if [[ -f "$_cfg_root/.env" ]]; then
  while IFS= read -r _line || [[ -n "$_line" ]]; do
    [[ "$_line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    _key=${BASH_REMATCH[2]}; _value=${BASH_REMATCH[3]}
    if [[ "$_value" =~ ^\"([^\"]*)\"[[:space:]]*(#.*)?$ || "$_value" =~ ^\'([^\']*)\'[[:space:]]*(#.*)?$ ]]; then
      _value=${BASH_REMATCH[1]}
    else
      _value=${_value%%#*}; _value=${_value%"${_value##*[![:space:]]}"}
    fi
    [[ -n "${!_key+set}" ]] || export "$_key=$_value"
  done < "$_cfg_root/.env"
fi
unset _n _line _key _value

# ---------------------------------------------------------------- the two Sparks
# This machine (the head) runs rank 0 and the API; WORKER (ssh target, key-based) runs rank 1. Two ranks only (--tp 2).
WORKER="${WORKER:-}"                   # e.g. user@<worker address>; set it in scripts/local.sh
FABRIC_PEER="${FABRIC_PEER:-}"         # the worker's CX7 address when WORKER is reached over another network
WORKER_HF_CACHE="${WORKER_HF_CACHE:-}" # the worker's Hugging Face cache when it is not its HF_HOME (absolute path)
MASTER_ADDR="${MASTER_ADDR:-}"         # the ranks' rendezvous (--master): default the head's address on the link
MASTER_PORT="${MASTER_PORT:-29500}"    # --master-port; keep it on the private link

# ---------------------------------------------------------------- the checkpoint and the EXL3 experts
# NVIDIA's FP4 checkpoint of Qwen3.8-Flash-Next, pinned. Both ranks serve exactly this snapshot from the local cache.
MODEL_ID="${MODEL_ID:-nvidia/Qwen3.8-Flash-Next-FP4-Local}"
MODEL_REVISION="${MODEL_REVISION:-b9ce18a6887e18c089fdea8509a33e63429a3300}"
CKPT_GIB=96                            # its 33 files at that revision: 95.24 GiB
# The routed experts come from turboderp's EXL3 quant (branch 2.68bpw_h6_ng8: 2.5 bpw, the edge layers 3.0), pinned to
# the branch's commit. tools/exl3_experts_pack.py takes only the routed-expert tensors (and the MTP head's) and lays them
# out per rank: rank 0 experts 0-255 plus the MTP head's, rank 1 experts 256-511, in experts-only safetensors files under
# HF_CACHE/EXL3_PACK, each checked against the sha256 in scripts/exl3-pack.sha256. EXL3_SOURCE=range fetches just those
# byte ranges of the pinned shards (~37.5 GiB); download fetches the six shards whole (~43.4 GiB, checked against the
# Hub's sha256) and cuts them locally. EXL3_EXPERTS=0 serves the checkpoint's own NVFP4 experts instead.
EXL3_EXPERTS="${EXL3_EXPERTS:-1}"
EXL3_REPO="${EXL3_REPO:-turboderp/Qwen3.8-Flash-Next-exl3}"
EXL3_BRANCH="${EXL3_BRANCH:-2.68bpw_h6_ng8}"
EXL3_REVISION="${EXL3_REVISION:-d660113325e1eba20e760a9063580139bdbdfc22}"
EXL3_SHARDS="${EXL3_SHARDS:-6}"
EXL3_PACK="${EXL3_PACK:-exl3-experts-2.68bpw}"
EXL3_SOURCE="${EXL3_SOURCE:-range}"
EXL3_LIMIT_RATE="${EXL3_LIMIT_RATE:-0}"   # curl --limit-rate for the range fetch (0: no limit), e.g. 50M

# ---------------------------------------------------------------- the engine and the image
# TensorFold at TF_REF (upstream main, 2026-11-02) with patches/*.patch applied in filename order (git apply in the
# checkout's root), built with Zig in NVIDIA's PyTorch container: `zig build -Doptimize=fast fatbins native`. The image
# also carries libjpeg-turbo and libwebp (image input) and the grammar library of TensorFold's v1.1.4 release, each
# pinned by sha256 (scripts/prepare.sh, the Dockerfile).
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"
TF_REF="${TF_REF:-089a18dff289d4c58043c1957ad0d7fe73d814b3}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
ZIG_VERSION="${ZIG_VERSION:-0.17.0}"
ZIG_SHA256="${ZIG_SHA256:-9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8}"   # zig-aarch64-linux-0.17.0.tar.xz
LIBJPEG_TURBO_VERSION="${LIBJPEG_TURBO_VERSION:-3.1.4.1}"
LIBJPEG_TURBO_SHA256="${LIBJPEG_TURBO_SHA256:-ecae8008e2cc9ade2f2c1bb9d5e6d4fb73e7c433866a056bd82980741571a022}"
LIBWEBP_VERSION="${LIBWEBP_VERSION:-1.6.0}"
LIBWEBP_SHA256="${LIBWEBP_SHA256:-e4ab7009bf0629fd11982d4c2aa83964cf244cffba7347ecd39019a9e38c4564}"
TF_RELEASE_VERSION="${TF_RELEASE_VERSION:-1.1.4}"
TF_RELEASE_AARCH64_SHA256="${TF_RELEASE_AARCH64_SHA256:-2c77ff0179dbfb9ae1c6ac647f85210194c151243f49eb93f0e34a8d53d7a4ec}"
# The image's hash: the patches, the Dockerfile (in prepare.sh) and every pinned input. A change makes a new tag, so
# start.sh asks prepare.sh to build it.
image_hash() {
  ( cd "$_cfg_root" || exit 1
    cat patches/*.patch 2>/dev/null
    sed -n "/<<'DOCKERFILE'/,/^DOCKERFILE\$/p" scripts/prepare.sh
    echo "$TF_REPO $TF_REF $BASE_IMAGE zig-$ZIG_VERSION $ZIG_SHA256 jpeg-$LIBJPEG_TURBO_VERSION $LIBJPEG_TURBO_SHA256" \
         "webp-$LIBWEBP_VERSION $LIBWEBP_SHA256 tf-$TF_RELEASE_VERSION $TF_RELEASE_AARCH64_SHA256"
  ) | sha256sum | cut -c1-12
}
IMAGE="${IMAGE:-tensorfold-qwen38-tp2:${TF_REF:0:9}-$(image_hash)}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-flash-next-tf}"   # the same name on both Sparks
# CUDA_COMPAT=1 puts the image's own CUDA forward-compatibility driver library (/usr/local/cuda/compat/lib.real) first on
# the engine's library path, ahead of the host's; 0 uses the host's libcuda.
CUDA_COMPAT="${CUDA_COMPAT:-1}"

# ---------------------------------------------------------------- serving
SERVED_NAME="${SERVED_NAME:-Qwen3.8-Flash-Next}"
HOST="${HOST:-0.0.0.0}"                # where rank 0's API listens (rank 1 serves no HTTP)
PORT="${PORT:-8000}"
CONTEXT="${CONTEXT:-409600}"           # prompt + reply window per stream (--context); past 262,144 it needs YaRN
PARALLEL="${PARALLEL:-24}"             # streams decoded together (--parallel)
KV_DTYPE="${KV_DTYPE:-nvfp4}"          # --kv-cache: nvfp4 (lossy), fp8 (lossy), bf16 (exact)
PROMPT_CACHE_GIB="${PROMPT_CACHE_GIB:-45}"   # kept conversation states for prefix reuse (--prompt-cache-gib)
MAX_TOKENS="${MAX_TOKENS:-32768}"      # reply budget of a request that sets no max_tokens (--max-tokens)
THINKING="${THINKING:-1}"              # 1: --thinking, 0: --no-thinking
REASONING_EFFORT="${REASONING_EFFORT:-xhigh}"   # --reasoning-effort, the default of a request that sets none
DRAFTS="${DRAFTS:-1}"                  # MTP drafts (0: --no-drafts)
TEMPERATURE="${TEMPERATURE:-1.0}"; TOP_P="${TOP_P:-0.95}"; TOP_K="${TOP_K:-20}"   # sampling defaults
VISION="${VISION:-1}"                  # image and video input (--vision)
MAX_IMAGES="${MAX_IMAGES:-50}"; MAX_VIDEOS="${MAX_VIDEOS:-4}"; IMAGE_TOKENS="${IMAGE_TOKENS:-16384}"
# The engine's own variables: scripts/engine.env, then any TENSORFOLD_*, TF_FLASHNEXT_* or TF_EXL3_* set above it (the
# environment, local.sh, .env) wins. ENGINE_UNSET lists names to leave out. TENSORFOLD_API_KEY goes to rank 0 by name
# only, never on a command line.
ENGINE_ENV_FILE="${ENGINE_ENV_FILE:-$_cfg_root/scripts/engine.env}"
ENGINE_UNSET="${ENGINE_UNSET:-}"

# ---------------------------------------------------------------- files, memory and timeouts
HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
KERNEL_CACHE="${KERNEL_CACHE:-$HOME/.cache/tensorfold-qwen38-tp2/cuda}"   # the ranks' /cache (CUDA's JIT cache)
STATE_DIR="${STATE_DIR:-$HOME/.local/state/qwen38-tp2-tensorfold}"   # locks and the setup marker
LOG_DIR="${LOG_DIR:-$HOME/.cache/tensorfold-qwen38-tp2/logs}"        # stop.sh saves each rank's log here first
LOG_KEEP="${LOG_KEEP:-10}"
# start.sh refuses a Spark with less MemAvailable than MEM_NEED_GIB before it launches (MEM_CHECK=0: a warning). The
# default is a guard chosen by this recipe, not a measurement of what a rank needs: run the pair on otherwise idle Sparks.
MEM_NEED_GIB="${MEM_NEED_GIB:-100}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-1800}"   # seconds start.sh waits for /health
STOP_TIMEOUT="${STOP_TIMEOUT:-30}"
IMAGE_FREE_GB="${IMAGE_FREE_GB:-40}"   # free disk prepare.sh asks for under Docker's root before it builds or loads
DOCKER_RUN_EXTRA="${DOCKER_RUN_EXTRA:-}"   # extra docker run arguments for both ranks (word-split), e.g. --cpuset-cpus 0-19

# Colours only on a terminal.
_c() { [[ -t "$1" ]] && printf '\033[%sm' "$2" || true; }
log()  { printf '%s[%s]%s %s\n' "$(_c 1 '1;36')" "$(basename "$0")" "$(_c 1 0)" "$*"; }
warn() { printf '%s[%s] WARN:%s %s\n' "$(_c 2 '1;33')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; }
die()  { printf '%s[%s] ERROR:%s %s\n' "$(_c 2 '1;31')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; exit 1; }

[[ -z "${HF_TOKEN:-}" ]] || export HF_TOKEN
model_dir() { echo "$HF_CACHE/hub/models--${MODEL_ID//\//--}/snapshots/$MODEL_REVISION"; }
pack_dir() { echo "$HF_CACHE/$EXL3_PACK"; }

# engine_env: the KEY=VALUE pairs both ranks get, one a line: scripts/engine.env with any value set above it, plus the
# TENSORFOLD_* / TF_FLASHNEXT_* / TF_EXL3_* names exported here that the file does not have; less ENGINE_UNSET.
# TF_FLASHNEXT_EXPERTS_EXL3 points at the pack (or is left out with EXL3_EXPERTS=0).
ENGINE_FORWARD='^(TENSORFOLD|TF_FLASHNEXT|TF_EXL3)_[A-Z0-9_]+$'
ENGINE_NCCL='^NCCL_(MAX_NCHANNELS|MIN_NCHANNELS|RUNTIME_CONNECT|PROTO|ALGO|BUFFSIZE|NTHREADS|NET_PLUGIN|NET_GDR_LEVEL|NET_GDR_C2C|DEBUG|DEBUG_SUBSYS)$'
# With EXL3_EXPERTS=0 the three EXL3 variables are left out. TF_FLASHNEXT_EXPERTS_TOPK=1..9 (the five-expert switch,
# README) leaves out TF_FLASHNEXT_EXPERT_REROUTE, which the engine refuses beside it, unless you set both (refused).
engine_env() {
  local line k v seen=" " drop=" $ENGINE_UNSET " topk=${TF_FLASHNEXT_EXPERTS_TOPK:-}
  [[ -r "$ENGINE_ENV_FILE" ]] || die "cannot read $ENGINE_ENV_FILE"
  [[ "$EXL3_EXPERTS" == 1 ]] || drop+="TF_FLASHNEXT_EXPERTS_EXL3 TF_FLASHNEXT_EXPERTS_EXL3_LAYERS TF_EXL3_PROMPT_ROWS "
  if [[ "$topk" =~ ^[1-9]$ ]]; then
    [[ -z "${TF_FLASHNEXT_EXPERT_REROUTE+set}" || "${TF_FLASHNEXT_EXPERT_REROUTE}" == 0 ]] ||
      die "TF_FLASHNEXT_EXPERTS_TOPK=$topk and TF_FLASHNEXT_EXPERT_REROUTE=$TF_FLASHNEXT_EXPERT_REROUTE: the engine refuses the two together"
    drop+="TF_FLASHNEXT_EXPERT_REROUTE "
  fi
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
    [[ "$line" =~ ^([A-Z][A-Z0-9_]*)=([A-Za-z0-9_.,:/+=-]*)$ ]] || die "$ENGINE_ENV_FILE: not KEY=VALUE with a plain value: ${line%%=*}"
    k=${BASH_REMATCH[1]}; v=${BASH_REMATCH[2]}
    [[ "$k" =~ $ENGINE_FORWARD || "$k" =~ $ENGINE_NCCL ]] || die "$ENGINE_ENV_FILE: $k is not an engine variable this recipe forwards"
    if [[ -n "${!k+set}" ]]; then v=${!k}
    elif [[ "$k" == TF_FLASHNEXT_EXPERTS_EXL3 ]]; then v=/root/.cache/huggingface/$EXL3_PACK; fi
    seen+="$k "
    [[ "$drop" == *" $k "* ]] && continue
    [[ "$v" =~ ^[A-Za-z0-9_.,:/+=-]*$ ]] || die "$k: not a plain value (letters, digits and _ . , : / + = -)"
    echo "$k=$v"
  done < "$ENGINE_ENV_FILE"
  while IFS= read -r k; do
    [[ "$k" =~ $ENGINE_FORWARD || "$k" =~ $ENGINE_NCCL ]] || continue
    [[ "$k" == TENSORFOLD_API_KEY || "$seen" == *" $k "* || "$drop" == *" $k "* ]] && continue
    [[ "${!k}" =~ ^[A-Za-z0-9_.,:/+=-]*$ ]] || die "$k: not a plain value (letters, digits and _ . , : / + = -)"
    echo "$k=${!k}"
  done < <(compgen -e | LC_ALL=C sort)
}

# What scripts/prepare.sh last left ready on both Sparks; start.sh runs it again when this differs.
PREPARED_MARKER="$STATE_DIR/prepared"
prepared_state() {
  echo "image=$IMAGE model=$MODEL_ID@$MODEL_REVISION exl3=$EXL3_EXPERTS:$EXL3_REPO@$EXL3_REVISION:$EXL3_PACK worker=$WORKER"
}
