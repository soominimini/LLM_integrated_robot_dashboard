#!/usr/bin/env bash
# Reload-aware watchdog for the QTrobot LLM web server.
# - Finds the worker dynamically (owner of :8080), so it survives auto-reloads.
# - A PID change is logged as a RELOAD (not an incident).
# - Real incidents -> exit 1 so the supervising session is notified:
#     * HTTP non-200 for >= FAIL_LIMIT consecutive checks
#     * a new Traceback / [ERROR] / CRITICAL appears in the trace log
# Writes a rolling heartbeat to scripts/watchdog.status.

DIR="/home/qtrobot/tutorials-master/demos/version_1_llm_gemini"
PORT=8080
TRACE="$DIR/src/logs/trace.log"
STATUS="$DIR/scripts/watchdog.status"
INTERVAL=10
FAIL_LIMIT=3                     # ~30s of HTTP failure before declaring an incident
ERR_RE='Traceback \(most recent call|\[ERROR\]|CRITICAL'

current_worker() {              # PID listening on :8080, or empty
  ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2
}

incident() { echo "INCIDENT $(date '+%H:%M:%S'): $1" | tee -a "$STATUS"; exit 1; }

last_pid="$(current_worker)"
base_err=$(grep -cE "$ERR_RE" "$TRACE" 2>/dev/null) || base_err=0
fails=0

while true; do
  ts=$(date '+%H:%M:%S')
  pid="$(current_worker)"

  # process presence (allow a brief gap during reload before alarming)
  if [ -z "$pid" ]; then
    fails=$((fails + 1))
    [ "$fails" -ge "$FAIL_LIMIT" ] && incident "no process listening on :$PORT for $((fails*INTERVAL))s"
    echo "$ts  WARN no-listener ($fails/$FAIL_LIMIT)" > "$STATUS"
    read -t "$INTERVAL" -u 9 _ 9</dev/null 2>/dev/null || true
    continue
  fi

  # reload detection (PID changed but server is back) -> re-baseline, not an incident
  if [ -n "$last_pid" ] && [ "$pid" != "$last_pid" ]; then
    echo "$ts  RELOAD worker $last_pid -> $pid" | tee -a "$STATUS"
    base_err=$(grep -cE "$ERR_RE" "$TRACE" 2>/dev/null) || base_err=0
  fi
  last_pid="$pid"

  # HTTP health
  code=$(curl -s -o /dev/null -m 5 -w '%{http_code}' "http://localhost:$PORT/" 2>/dev/null)
  if [ "$code" != "200" ]; then
    fails=$((fails + 1))
    [ "$fails" -ge "$FAIL_LIMIT" ] && incident "HTTP :$PORT = '$code' for $((fails*INTERVAL))s (worker $pid)"
  else
    fails=0
  fi

  # new errors in the trace
  now_err=$(grep -cE "$ERR_RE" "$TRACE" 2>/dev/null) || now_err=0
  if [ "$now_err" -gt "$base_err" ]; then
    line=$(grep -nE "$ERR_RE" "$TRACE" | tail -1)
    incident "new error in trace: $line"
  fi

  rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')
  reqs=$(grep -c '┌─ WORK' "$TRACE" 2>/dev/null) || reqs=0
  echo "$ts  OK  worker=$pid http=$code rss=${rss}KB works_traced=$reqs" > "$STATUS"

  read -t "$INTERVAL" -u 9 _ 9</dev/null 2>/dev/null || true
done
