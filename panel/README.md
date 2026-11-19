# Веб-панель (статическая)

Онлайн-версия: **https://palcikia-crypto.github.io/pseudo-opt-bot/**

Два источника:
- сайт-оболочка — деплоится воркфлоу `panel-pages.yml` (GitHub Pages) или
  `cloudflare-pages.yml` (Cloudflare Pages, проект `pseudo-opt-panel`);
- данные — снапшот `state.json` из ветки `panel-site`, который публикует
  `panel/snapshot.py` там, где работает бот (`sh panel/sync.sh` — цикл),
  плюс live-цены Bybit, которые браузер запроса́ет напрямую.

Локальная копия панели (читает `paper_state.json` прямо с диска):
`DASH_PORT=8080 sh run_bot.sh` → http://127.0.0.1:8080
