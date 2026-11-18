#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BINARY="$ROOT/bin/kernel-sandbox-master"
RUN_DIR=""
PID_FILE=""
LOG_FILE=""

set_run_files() {
  local config=$1 master_id
  master_id=$(awk '$1 == "master_id:" {print $2; exit}' "$config")
  [[ -n "$master_id" ]] || { echo "master_id is missing from $config" >&2; exit 2; }
  RUN_DIR="${KERNEL_SANDBOX_RUN_DIR:-$ROOT/.run/$master_id}"
  PID_FILE="$RUN_DIR/master.pid"
  LOG_FILE="$RUN_DIR/master.log"
}

is_master_pid() {
  local pid=$1 command
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null || return 1
  [[ $(readlink -f "/proc/$pid/exe" 2>/dev/null) == "$BINARY" ]] && return 0
  IFS= read -r -d '' command <"/proc/$pid/cmdline" 2>/dev/null || true
  [[ "$command" == "$BINARY" ]]
}

running() {
  [[ -f "$PID_FILE" ]] && is_master_pid "$(cat "$PID_FILE")"
}

master_pid() {
  if running; then
    cat "$PID_FILE"
    return
  fi
  return 0
}

build() {
  make -C "$ROOT" build
}

start() {
  [[ $# -eq 1 ]] || { echo "usage: $0 start <config-file>" >&2; exit 2; }
  local config
  config=$(realpath "$1")
  set_run_files "$config"
  running && { echo "Master is already running (PID $(cat "$PID_FILE"))"; return; }
  [[ -x "$BINARY" ]] || build
  mkdir -p "$RUN_DIR"
  (
    cd "$ROOT"
    nohup "$BINARY" --config "$config" >>"$LOG_FILE" 2>&1 &
    echo $! >"$PID_FILE"
  )
  sleep 1
  running || { echo "Master failed to start; see $LOG_FILE" >&2; exit 1; }
  echo "Master started (PID $(cat "$PID_FILE")); log: $LOG_FILE"
}

stop() {
  set_run_files "$(realpath "$1")"
  local pid
  pid=$(master_pid)
  if [[ -z "$pid" ]]; then
    rm -f "$PID_FILE"
    echo "Master is not running"
    return
  fi
  kill "$pid"
  for _ in {1..50}; do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.1
  done
  kill -0 "$pid" 2>/dev/null && { echo "Master did not stop (PID $pid)" >&2; exit 1; }
  rm -f "$PID_FILE"
  echo "Master stopped"
}

case "${1:-}" in
  build)
    [[ $# -eq 1 ]] || { echo "usage: $0 build" >&2; exit 2; }
    build
    ;;
  start)
    shift
    start "$@"
    ;;
  stop)
    [[ $# -eq 2 ]] || { echo "usage: $0 stop <config-file>" >&2; exit 2; }
    stop "$2"
    ;;
  restart)
    [[ $# -eq 2 ]] || { echo "usage: $0 restart <config-file>" >&2; exit 2; }
    config=$2
    stop "$config"
    build
    start "$config"
    ;;
  status)
    [[ $# -eq 2 ]] || { echo "usage: $0 status <config-file>" >&2; exit 2; }
    set_run_files "$(realpath "$2")"
    running && echo "Master is running (PID $(cat "$PID_FILE"))" || { echo "Master is not running"; exit 1; }
    ;;
  *)
    echo "usage: $0 build|start <config-file>|stop <config-file>|restart <config-file>|status <config-file>" >&2
    exit 2
    ;;
esac
