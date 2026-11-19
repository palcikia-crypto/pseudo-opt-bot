#!/bin/sh
# sync.sh — циклическая публикация снапшота бота на ветку PANEL_BRANCH (для Pages).
# Пример:  PANEL_SYNC_SEC=600 PANEL_BRANCH=panel-site sh panel/sync.sh
cd "$(dirname "$0")/.."
INTERVAL="${PANEL_SYNC_SEC:-600}"
PY="./venv/bin/python"
[ -x "$PY" ] || PY="python3"

while true; do
  if [ -f "${PAPER_STATE_FILE:-paper_state.json}" ]; then
    PAPER_STATE_FILE="${PAPER_STATE_FILE:-paper_state.json}" \
    PANEL_BRANCH="${PANEL_BRANCH:-panel-site}" "$PY" panel/snapshot.py >> panel.log 2>&1 \
      && echo "[SYNC] ok ($(date -u '+%FT%TZ'))" || echo "[SYNC] FAIL ($(date -u '+%FT%TZ'))"
  else
    echo "[SYNC] нет ${PAPER_STATE_FILE:-paper_state.json} — жду"
  fi
  sleep "$INTERVAL"
done
