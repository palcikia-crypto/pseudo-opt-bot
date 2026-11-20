#!/usr/bin/env bash
# research/watchdog.sh — сторож всех демонов (раунд 9d).
# Проверяет каждые 60с, живы ли процессы; мёртвый — поднимает заново
# (тот же рабочий каталог, тот же state — все процессы stateful/resume-able:
#  paper.py и evo/optevo/optpaper восстанавливаются из json-состояния).
# После сбоя песочницы достаточно одной команды:
#   (setsid nohup bash research/watchdog.sh > research/watchdog.log 2>&1 &)
# Проверка одного цикла: bash research/watchdog.sh --once
cd "$(dirname "$0")/.." || exit 1

alarm() { echo "[watchdog $(date -u +%H:%M:%S)] $*"; }

ensure() {
  local name="$1"; shift
  local pattern="$1"; shift
  if pgrep -f "$pattern" > /dev/null 2>&1; then
    return 0
  fi
  alarm "поднимаю $name"
  (setsid nohup "$@" > "research/$name.log" 2>&1 < /dev/null &)
}

main() {
  # Основной бот — супервизор run_bot.sh (сам перезапускает paper.py).
  ensure bot      '[r]un_bot.sh'   bash run_bot.sh
  ensure paper_deep '[p]aper_deep.py' ./venv/bin/python research/paper_deep.py
  ensure loblog   '[l]oblog.py'    ./venv/bin/python research/loblog.py
  ensure optpaper '[o]ptpaper.py'  ./venv/bin/python research/optpaper.py
  ensure panelsync '[p]anel/sync.sh' bash panel/sync.sh
  ensure evo      '[e]vo.py --daemon' ./venv/bin/python research/evo.py --daemon \
      --generations 30 --pop 40 --cycle-sec 1800 \
      --symbols BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,APTUSDT,SEIUSDT,DOGEUSDT,BNBUSDT,AVAXUSDT,NEARUSDT
  ensure optevo   '[o]ptevo.py --daemon' ./venv/bin/python research/optevo.py --daemon \
      --generations 25 --pop 30 --cycle-sec 2400
}

case "$1" in
  --once)
    main
    echo "[watchdog] цикл проверки пройден (--once)"
    ;;
  *)
    while true; do
      main
      sleep 60
    done
    ;;
esac
