#!/usr/bin/env bash
# Maintenance window for a model that needs the whole GPU (here: Strata Qwen3.8-Flash-Next IQ3_S on :8090).
# 1. pause the production llama-server (Bonsai on :8081), 2. start the server under test, 3. run medbench (resumable),
# 4. ALWAYS stop the test server and restore production, also after an error, Ctrl-C or SIGTERM.
#
#   PARALLEL=3 scripts/strata_window.sh configs/strata-flashnext-iq3s-medium.json [extra medbench run args...]
#
# Environment (defaults match the llm01 server this benchmark was run on):
#   PROD_REPO     llmapi_server checkout with scripts/start_bonsai.sh and config/bonsai.pause   [~/llmapi_server]
#   PROD_PATTERN  command-line pattern of the production llama-server                           [Ternary-Bonsai-2-27B]
#   PROD_URL      production health URL                                                         [http://127.0.0.1:8081/health]
#   STRATA_HOME   Strata Terminal folder (serve/server.py, python/)                             [~/.local/share/strata/terminal/0.1.45.3]
#   STRATA_CONFIG Strata run config (exe, args, tokenizer, port)                                [~/.local/share/strata/install/strata-iq3_s.json]
#   PARALLEL      simultaneous requests in the engine (Strata --parallel; 1 = one at a time)   [3]
#   LOG_DIR       logs                                                                          [results/logs]
set -u
CFG="${1:?usage: strata_window.sh <medbench config> [medbench run args...]}"
shift
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PROD_REPO="${PROD_REPO:-$HOME/llmapi_server}"
PROD_PATTERN="${PROD_PATTERN:-Ternary-Bonsai-2-27B}"
PROD_URL="${PROD_URL:-http://127.0.0.1:8081/health}"
STRATA_HOME="${STRATA_HOME:-$HOME/.local/share/strata/terminal/0.1.45.3}"
STRATA_CONFIG="${STRATA_CONFIG:-$HOME/.local/share/strata/install/strata-iq3_s.json}"
PARALLEL="${PARALLEL:-3}"
LOG_DIR="${LOG_DIR:-$HERE/results/logs}"
PAUSE="$PROD_REPO/config/bonsai.pause"
URL="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["base_url"].rstrip("/"))' "$CFG")"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
STRATA_LOG="$LOG_DIR/strata-$STAMP.log"
log() { echo "[window $(date -u +%H:%M:%S)] $*"; }

prod_systemd() { [ "$(systemctl show -p ActiveState --value bonsai 2>/dev/null)" = active ]; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' '; }

STRATA_PID=""
restore() {
  trap - EXIT INT TERM
  log "stopping the server under test"
  [ -n "$STRATA_PID" ] && kill "$STRATA_PID" 2>/dev/null
  for _ in $(seq 30); do pgrep -f "strata-engine/x/strata" >/dev/null || break; sleep 1; done
  pkill -9 -f "strata-engine/x/strata" 2>/dev/null
  [ -n "$STRATA_PID" ] && kill -9 "$STRATA_PID" 2>/dev/null
  log "restoring production"
  rm -f "$PAUSE"
  if ! prod_systemd && ! pgrep -f "$PROD_PATTERN" >/dev/null; then
    setsid nohup bash "$PROD_REPO/scripts/start_bonsai.sh" >> "$HOME/.local/share/bonsai-manual.log" 2>&1 < /dev/null &
  fi
  for _ in $(seq 120); do
    if curl -sf -m 3 "$PROD_URL" >/dev/null; then log "production healthy again"; return 0; fi
    sleep 5
  done
  log "WARNING: production did not become healthy within 10 min; check $PROD_REPO"
  return 1
}
trap restore EXIT
trap 'exit 130' INT TERM

log "pausing production ($PROD_PATTERN)"
touch "$PAUSE"
PIDS="$(pgrep -f "$PROD_PATTERN" | tr '\n' ' ')"
for p in $PIDS; do
  # SIGKILL: a clean exit would leave a systemd unit with Restart=on-failure stopped; this way it restarts and waits on the pause file
  [ "$(cat /proc/$p/comm 2>/dev/null)" = llama-server ] && kill -9 "$p"
done
before="$(gpu_used)"
for _ in $(seq 60); do
  pgrep -f "$PROD_PATTERN" >/dev/null || break
  sleep 1
done
sleep 3
log "GPU memory used: ${before} -> $(gpu_used) MiB"

log "starting Strata (parallel=$PARALLEL) -> $STRATA_LOG"
PAR_ARGS=()
[ "$PARALLEL" -gt 1 ] && PAR_ARGS=(--parallel "$PARALLEL")
cd "$STRATA_HOME"
setsid "$STRATA_HOME/python/bin/python3.12" serve/server.py --engine strata --config "$STRATA_CONFIG" \
  --port "${URL##*:}" "${PAR_ARGS[@]}" >> "$STRATA_LOG" 2>&1 < /dev/null &
STRATA_PID=$!
cd "$HERE"
for _ in $(seq 120); do curl -sf -m 3 "$URL/health" >/dev/null && break; sleep 5; done
log "warm-up request (loads the model)"
curl -sf -m 900 "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say OK."}],"max_tokens":16,"reasoning_effort":"none"}' >/dev/null \
  || { log "Strata did not answer; see $STRATA_LOG"; exit 1; }
grep -E "batch_slots|parallel requests|expert.cache|RAM" "$STRATA_LOG" | tail -5
log "GPU memory used with Strata: $(gpu_used) MiB"

log "medbench run $CFG $*"
"$HERE/.venv/bin/medbench" --data "$HERE/data" run --config "$CFG" "$@"
rc=$?
log "medbench exited with $rc"
exit $rc
