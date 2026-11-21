#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next from two DGX Sparks with TensorFold at --tp 2: rank 1 on the worker (WORKER), rank 0 here,
# which serves the OpenAI-compatible API. Runs scripts/prepare.sh first when the image or the weights are not ready on
# both Sparks, waits for /health, then sends one smoke request. Stop it with ./stop.sh.
# Adapted from MiaAI-Lab's Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold start.sh (Apache-2.0; CREDITS.md).
#
# Usage: ./start.sh [restart] [extra tensorfold-native serve arguments]
#   ./start.sh                  the defaults (scripts/config.sh, scripts/engine.env): 24 streams x 409,600 tokens,
#                               NVFP4 KV cache, a 45 GiB prompt cache, EXL3 routed experts, MTP drafts, image input
#   ./start.sh restart          stop both ranks (./stop.sh), then start them again (new settings or patches)
#   DRY_RUN=1 ./start.sh        print both ranks' docker commands and the engine variables; start nothing
#   TF_FLASHNEXT_EXPERTS_TOPK=5 ./start.sh restart   five experts a token instead of ten (lower quality; README)
# Extra arguments go to both ranks after the defaults, so they win (the last value of a flag counts).
# Settings come from the environment, scripts/local.sh or ./.env (defaults in scripts/config.sh):
#   serving  CONTEXT, PARALLEL, KV_DTYPE, PROMPT_CACHE_GIB, MAX_TOKENS, THINKING, REASONING_EFFORT, DRAFTS,
#            TEMPERATURE, TOP_P, TOP_K, VISION, MAX_IMAGES, MAX_VIDEOS, IMAGE_TOKENS, SERVED_NAME, HOST, PORT
#   engine   scripts/engine.env (every TENSORFOLD_* / TF_FLASHNEXT_* / TF_EXL3_* variable; yours win), ENGINE_UNSET,
#            EXL3_EXPERTS, CUDA_COMPAT, TENSORFOLD_API_KEY (by name, to rank 0 only)
#   nodes    WORKER, FABRIC_PEER, WORKER_HF_CACHE, MASTER_ADDR, MASTER_PORT, NCCL_RAILS=1 (one RoCE device)
#   setup    PREPARE (auto | 1 | 0), MEM_NEED_GIB, MEM_CHECK=0, WAIT_TIMEOUT, DOCKER_RUN_EXTRA, LOG_DIR, LOG_KEEP
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./scripts/config.sh
source ./scripts/nodes.sh

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
MODE=start
case "${1:-}" in
  restart) MODE=restart; shift ;;
  help) usage; exit 0 ;;
esac
for arg in "$@"; do [[ "$arg" == -h || "$arg" == --help ]] && { usage; exit 0; }; done
USER_ARGS=("$@")
DRY=0; [[ "${DRY_RUN:-0}" == 1 ]] && DRY=1

# ---------------------------------------------------------------- settings
check_workers
num() { [[ "${!1}" =~ ^[1-9][0-9]*$ ]] || die "$1 is a positive whole number, not ${!1}"; }
for v in CONTEXT PARALLEL PROMPT_CACHE_GIB MAX_TOKENS MAX_IMAGES MAX_VIDEOS IMAGE_TOKENS PORT MASTER_PORT TOP_K MEM_NEED_GIB; do num "$v"; done
for v in THINKING DRAFTS VISION CUDA_COMPAT EXL3_EXPERTS; do [[ "${!v}" =~ ^[01]$ ]] || die "$v is 0 or 1, not ${!v}"; done
[[ "$KV_DTYPE" =~ ^(nvfp4|fp8|bf16)$ ]] || die "KV_DTYPE is nvfp4, fp8 or bf16, not $KV_DTYPE"
[[ "$REASONING_EFFORT" =~ ^[a-z]+$ ]] || die "REASONING_EFFORT is a word such as low, medium, high or xhigh"
[[ "$TEMPERATURE" =~ ^[0-9]+(\.[0-9]+)?$ && "$TOP_P" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "TEMPERATURE and TOP_P are numbers"
[[ "$HOST" =~ ^[0-9A-Za-z.:-]+$ && "$SERVED_NAME" =~ ^[A-Za-z0-9_.:/+-]+$ ]] || die "HOST or SERVED_NAME holds characters this script does not pass on"
ENGINE_LINES=$(engine_env) || exit 1
mapfile -t ENGINE_KV <<<"$ENGINE_LINES"
engine_value() { local l; for l in "${ENGINE_KV[@]}"; do [[ "$l" == "$1="* ]] && { echo "${l#*=}"; return 0; }; done; return 0; }
# two ranks refuse a partly kept window's late DeltaNet keep of one rank (TF_FLASHNEXT_KEEP_LATE, not TP_KEEP_LATE)
[[ -z "$(engine_value TF_FLASHNEXT_KEEP_LATE)" || "$(engine_value TF_FLASHNEXT_KEEP_LATE)" == 0 ]] ||
  die "TF_FLASHNEXT_KEEP_LATE is refused at --tp 2 (unset it or set 0)"
yarn=$(engine_value TF_FLASHNEXT_YARN)
(( CONTEXT <= 262144 )) || [[ "$yarn" =~ ^[2-4]$ ]] ||
  die "CONTEXT=$CONTEXT is past the trained 262,144 tokens: it needs TF_FLASHNEXT_YARN (2 to 4; engine.env sets 2)"
(( CONTEXT <= 262144 * ${yarn:-1} )) || die "CONTEXT=$CONTEXT is past TF_FLASHNEXT_YARN=$yarn x 262,144 tokens"

# ---------------------------------------------------------------- one start at a time, the worker, a running server
command -v docker >/dev/null || die "docker is not installed"
mkdir -p "$KERNEL_CACHE" "$STATE_DIR"
exec 8>"$STATE_DIR/start.lock"
flock -n 8 || die "another ./start.sh is already running; wait for it to finish"
need_worker 1
(( DRY )) && log "DRY_RUN=1: printing the docker commands; nothing is stopped, started or prepared"
running_here()   { [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; }
running_worker() { [[ "$(worker 1 docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; }
API_HOST="$HOST"; [[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] && API_HOST=127.0.0.1
[[ "$API_HOST" == *:* ]] && API_HOST="[$API_HOST]"
URL="http://$API_HOST:$PORT"
if (( ! DRY )) && [[ "$MODE" == start ]] && running_here && running_worker; then
  log "$CONTAINER_NAME already runs on both Sparks (port $PORT): ./start.sh restart restarts it, ./stop.sh stops it"
  exit 0
fi

# ---------------------------------------------------------------- 1. setup
if [[ "${PREPARE:-auto}" == 1 || ( "${PREPARE:-auto}" != 0 && "$(prepared_state)" != "$(cat "$PREPARED_MARKER" 2>/dev/null)" ) ]]; then
  if (( DRY )); then log "DRY_RUN: scripts/prepare.sh would run now (the setup is not ready, or PREPARE=1)"
  else log "Setup not ready on both Sparks: running scripts/prepare.sh"; ./scripts/prepare.sh; fi
fi
missing() { if (( DRY )); then warn "$1"; else die "$1"; fi; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || missing "image $IMAGE missing here: run scripts/prepare.sh"
[[ -n "$WORKER_DOWN" ]] || worker 1 docker image inspect "$IMAGE" >/dev/null 2>&1 || missing "image $IMAGE missing on the worker: run scripts/prepare.sh"
WORKER_HF=$(worker_hf_cache)
SUB="hub/models--${MODEL_ID//\//--}/snapshots/$MODEL_REVISION"
SNAP="/root/.cache/huggingface/$SUB"
[[ -s "$HF_CACHE/$SUB/config.json" ]] || missing "checkpoint $MODEL_ID @ ${MODEL_REVISION:0:8} missing here: run scripts/prepare.sh"
[[ -n "$WORKER_DOWN" ]] || worker 1 "test -s '$WORKER_HF/$SUB/config.json'" || missing "checkpoint missing on the worker: run scripts/prepare.sh"
if [[ "$EXL3_EXPERTS" == 1 ]]; then
  [[ -s "$(pack_dir)/model.safetensors.index.json" && -f "$(pack_dir)/.ok-head" ]] || missing "EXL3 experts missing here ($(pack_dir)): run scripts/prepare.sh"
  [[ -n "$WORKER_DOWN" ]] || worker 1 "test -f '$WORKER_HF/$EXL3_PACK/.ok-worker'" || missing "rank 1's EXL3 experts missing on the worker: run scripts/prepare.sh"
fi

# ---------------------------------------------------------------- 2. checks: link, previous server, port, memory
detect_link
log "Link: $HEAD_ADDR ($HEAD_DEV) <-> $WORKER_ADDR ($WORKER_DEV), RoCE $HEAD_HCAS / $WORKER_HCAS, rendezvous $MASTER_ADDR:$MASTER_PORT"
if (( ! DRY )); then
  if running_here || running_worker; then ./stop.sh; fi
  # a stopped container of the same name (a crash, or an earlier run) is removed, its log saved first
  if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    saved=$(save_log "$LOG_DIR" 0 "$CONTAINER_NAME" "$LOG_KEEP") || warn "could not save rank 0's earlier log"
    [[ -z "${saved:-}" ]] || log "Rank 0's earlier log: $saved"
    docker rm -f "$CONTAINER_NAME" >/dev/null
  fi
  if worker 1 "docker ps -a --format '{{.Names}}' | grep -qx '$CONTAINER_NAME'"; then
    saved=$(worker_save_log) || warn "could not save rank 1's earlier log"
    [[ -z "${saved:-}" ]] || log "Rank 1's earlier log (on the worker): $saved"
    worker 1 "docker rm -f '$CONTAINER_NAME' >/dev/null"
  fi
  ! ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN || die "port $PORT is in use here"
  ! worker 1 "ss -ltn 'sport = :$PORT' | grep -q LISTEN" || die "port $PORT is in use on the worker (rank 1 binds it on 127.0.0.1)"
fi
here_gb=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
there_gb=$(worker 1 "awk '/MemAvailable/ {print int(\$2 / 1048576)}' /proc/meminfo" 2>/dev/null || echo "?")
for m in "here:$here_gb" "on the worker:$there_gb"; do
  g=${m#*:}; [[ "$g" =~ ^[0-9]+$ ]] || continue
  if (( g < MEM_NEED_GIB )); then
    msg="only $g GiB MemAvailable ${m%%:*} (MEM_NEED_GIB=$MEM_NEED_GIB): stop other GPU work there first"
    if (( DRY )) || [[ "${MEM_CHECK:-1}" == 0 ]]; then warn "$msg"; else die "$msg (MEM_CHECK=0 starts anyway)"; fi
  fi
done
log "Memory: $here_gb GiB MemAvailable here, $there_gb GiB on the worker"

# ---------------------------------------------------------------- 3. the two ranks' commands
SERVE=(serve "$SNAP" --tp 2 --master "$MASTER_ADDR" --master-port "$MASTER_PORT" --name "$SERVED_NAME"
       --context "$CONTEXT" --parallel "$PARALLEL" --kv-cache "$KV_DTYPE" --prompt-cache-gib "$PROMPT_CACHE_GIB"
       --max-tokens "$MAX_TOKENS" --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K"
       --reasoning-effort "$REASONING_EFFORT" --no-update-check)
if [[ "$THINKING" == 1 ]]; then SERVE+=(--thinking); else SERVE+=(--no-thinking); fi
[[ "$DRAFTS" == 1 ]] || SERVE+=(--no-drafts)
[[ "$VISION" == 1 ]] && SERVE+=(--vision --vision-max-images "$MAX_IMAGES" --vision-max-videos "$MAX_VIDEOS" --vision-image-tokens "$IMAGE_TOKENS")
# both ranks see the same checkpoint identity without hashing ~96 GiB at every start: the repository id and revision
ident=$(printf '%s@%s' "$MODEL_ID" "$MODEL_REVISION" | sha256sum | cut -c1-64)
ENVS=(-e HF_HUB_OFFLINE=1 -e "TENSORFOLD_CHECKPOINT_SHA256=$ident" -e BASH_ENV= -e ENV=)
for e in "${ENGINE_KV[@]}"; do [[ -n "$e" ]] && ENVS+=(-e "$e"); done
if [[ "$CUDA_COMPAT" == 1 ]]; then
  img_ldp=$(docker image inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$IMAGE" 2>/dev/null | sed -n 's/^LD_LIBRARY_PATH=//p' | tail -1 || true)
  [[ "$img_ldp" =~ ^[A-Za-z0-9_.:/+-]*$ ]] || die "$IMAGE: its LD_LIBRARY_PATH holds characters this script does not pass on"
  ENVS+=(-e "LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real${img_ldp:+:$img_ldp}")
fi
KEY_ARGS=(); [[ -z "${TENSORFOLD_API_KEY:-}" ]] || { export TENSORFOLD_API_KEY; KEY_ARGS=(-e TENSORFOLD_API_KEY); }
read -r -a EXTRA <<<"$DOCKER_RUN_EXTRA"
COMMON=(--gpus all --ipc=host --network host --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK
        --device /dev/infiniband --restart no --log-opt max-size=200m --log-opt max-file=3 "${EXTRA[@]}")
APP=/opt/tensorfold/native/bin/tensorfold-native
read -r -a NCCL0 <<<"$(rank_nccl_env 0)"
read -r -a NCCL1 <<<"$(rank_nccl_env 1)"
cmd0=(docker run -d --name "$CONTAINER_NAME" "${COMMON[@]}" -v "$HF_CACHE:/root/.cache/huggingface:ro"
      -v "$KERNEL_CACHE:/cache" "${NCCL0[@]}" "${ENVS[@]}" "${KEY_ARGS[@]}" --entrypoint "$APP" "$IMAGE"
      "${SERVE[@]}" --rank 0 --host "$HOST" --port "$PORT" "${USER_ARGS[@]}")
cmd1=(docker run -d --name "$CONTAINER_NAME" "${COMMON[@]}" -v "$WORKER_HF:/root/.cache/huggingface:ro"
      -v "\$HOME/.cache/tensorfold-qwen38-tp2/cuda:/cache" "${NCCL1[@]}" "${ENVS[@]}" --entrypoint "$APP" "$IMAGE"
      "${SERVE[@]}" --rank 1 --host 127.0.0.1 --port "$PORT" "${USER_ARGS[@]}")
remote="mkdir -p \$HOME/.cache/tensorfold-qwen38-tp2/cuda &&"
for a in "${cmd1[@]}"; do case "$a" in '$HOME'*) remote+=" \"$a\"" ;; *) remote+=" $(printf '%q' "$a")" ;; esac; done
if (( DRY )); then
  printf '[dry-run] rank 1 on %s:\n  %s\n' "$WORKER" "$remote"
  printf '[dry-run] rank 0 here:\n  %s\n' "$(printf '%q ' "${cmd0[@]}" | sed 's/ $//')"
  printf '[dry-run] %s engine variables to both ranks\n' "${#ENGINE_KV[@]}"
  exit 0
fi

# ---------------------------------------------------------------- 4. launch and wait
log "Rank 1 on $WORKER"
worker 1 "$remote" >/dev/null || die "could not start rank 1 on $WORKER"
log "Rank 0 here: tensorfold-native ${SERVE[*]} --rank 0 --host $HOST --port $PORT ${USER_ARGS[*]}"
"${cmd0[@]}" >/dev/null
fail() {
  printf '\n── rank 0 (here): last log lines ──\n'; docker logs --tail 25 "$CONTAINER_NAME" 2>&1 | sed 's/^/  │ /'
  printf '── rank 1 (%s): last log lines ──\n' "$WORKER"; worker 1 docker logs --tail 25 "$CONTAINER_NAME" 2>&1 | sed 's/^/  │ /'
  die "$1"
}
s=$SECONDS; next=30
until curl -sf --max-time 5 "$URL/health" >/dev/null 2>&1; do
  running_here || fail "rank 0 exited (code $(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER_NAME")) before it was ready"
  (( SECONDS - s < WAIT_TIMEOUT )) || fail "no /health after ${WAIT_TIMEOUT}s (WAIT_TIMEOUT); the ranks still run: docker logs -f $CONTAINER_NAME"
  if (( SECONDS - s >= next )); then
    running_worker || fail "rank 1 on $WORKER exited before the server was ready"
    log "loading: $((SECONDS - s))s, $(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo) GiB MemAvailable here"
    next=$((next + 30))
  fi
  sleep 5
done
log "Healthy after $((SECONDS - s))s"

# ---------------------------------------------------------------- 5. smoke test
body="{\"model\": \"$SERVED_NAME\", \"max_tokens\": 32, \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false}, \"messages\": [{\"role\": \"user\", \"content\": \"Reply with OK.\"}]}"
smoke() {
  if [[ -n "${TENSORFOLD_API_KEY:-}" ]]; then
    printf 'Authorization: Bearer %s\n' "$TENSORFOLD_API_KEY" |
      curl -sf --max-time 180 "$URL/v1/chat/completions" -H 'Content-Type: application/json' -H @- -d "$body"
  else
    curl -sf --max-time 180 "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$body"
  fi
}
reply=$(smoke | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip()[:40])') ||
  fail "the smoke request failed; the ranks still run"
[[ -n "$reply" ]] || fail "the smoke reply was empty; the ranks still run"
log "Smoke reply: $reply"
log "$SERVED_NAME is serving: $URL/v1 ($PARALLEL streams x $CONTEXT tokens). Logs: docker logs -f $CONTAINER_NAME; stop: ./stop.sh"
