#!/bin/sh
# run_bot.sh — paper-бот + веб-панель с автоперезапуском.
# Пример: DASH_HOST=0.0.0.0 DASH_PORT=8080 sh run_bot.sh
cd "$(dirname "$0")"
DASH_PORT="${DASH_PORT:-8080}"
DASH_HOST="${DASH_HOST:-127.0.0.1}"
PAPER_STATE_FILE="${PAPER_STATE_FILE:-paper_state.json}"
PAPER_POLL_SEC="${PAPER_POLL_SEC:-10}"
PY="./venv/bin/python"
[ -x "$PY" ] || PY="python3"

while true; do
  echo "[SUPERVISOR] старт paper-бота ($(date -u '+%FT%TZ'))"
  DASH_PORT="$DASH_PORT" DASH_HOST="$DASH_HOST" \
  PAPER_STATE_FILE="$PAPER_STATE_FILE" PAPER_POLL_SEC="$PAPER_POLL_SEC" \
    "$PY" -u paper.py >> paper.log 2>&1
  echo "[SUPERVISOR] бот завершился, перезапуск через 5 с ($(date -u '+%FT%TZ'))"
  sleep 5
done
