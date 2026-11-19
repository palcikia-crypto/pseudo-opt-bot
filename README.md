# ⚡ Pseudo-Options Scalper (Bybit USDT-Perpetuals)

Бот торгует ликвидные бессрочные USDT-фьючерсы Bybit на 1m/5m в режиме «опционного
билета»: фиксированный риск на сделку, SL/TP на стороне биржи, RR 2.5:1, вход
лимитным postOnly-ордером (maker-комиссия).

## Структура
| Файл | Назначение |
|---|---|
| `config.py` | настройки из `.env` |
| `exchange.py` | async-обертка Bybit v5 (ccxt) + PaperExchange (без ключей) |
| `indicators.py` | EMA/BB/RSI/ATR/VolSMA + дисбаланс стакана (numpy) |
| `strategy.py` | фильтр волатильности, тренд 5m, импульс 1m, объём, RSI, стакан |
| `risk_manager.py` | лот под риск, лимит позиций, дневной стоп-аут (state.json) |
| `trader.py` | postOnly вход с таймаутом, биржевые SL/TP, безубыток, тайм-стоп |
| `main.py` | asyncio-цикл: сигналы на закрытии 1m, монитор каждые 2 с |
| `paper.py` | live paper-trading на публичных данных Bybit — ключи НЕ нужны |
| `dashboard.py` | веб-панель: equity, PnL, открытые позиции (live-цены Bybit), история сделок, WR/PF, статус стоп-аута |
| `run_bot.sh` | супервизор-цикл: бот + панель с автоперезапуском |
| `tests/local_check.py` | офлайн-проверка математики без сети |

## Быстрый старт (Ubuntu VPS)
```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip
git clone https://github.com/palcikia-crypto/pseudo-opt-bot.git && cd pseudo-opt-bot
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # заполнить BYBIT_API_KEY/SECRET, TESTNET=true
python tests/local_check.py
python main.py
```

## Веб-панель
Онлайн: **https://palcikia-crypto.github.io/pseudo-opt-bot/** (GitHub Pages, деплой через
.github/workflows/panel-pages.yml) — показывает снапшот из ветки `panel-site` +
live-цены Bybit в браузере. Обновление снапшота там, где работает бот:
`PANEL_SYNC_SEC=600 sh panel/sync.sh`.

Локально:
```bash
sh run_bot.sh                 # бот + панель http://127.0.0.1:8080
DASH_HOST=0.0.0.0 sh run_bot.sh   # на VPS: панель по IP (закройте фаерволом/SSH-туннелем!)
```
Панель (`/`) обновляется каждые 3 с, JSON-эндпоинты: `/api/summary`, `/api/health`.
Отдельно (без бота, только чтение состояния): `python dashboard.py`.

## Paper-режим (реальные данные, без ключей)
```bash
python paper.py
```
Читает реальные 1m/5m свечи и стакан Bybit, симулирует сделки с реальными
комиссиями (maker 0.02% / taker 0.055%), безубытком, тайм-стопом и дневным
стопом. Состояние: `paper_state.json`. Идеально для проверки стратегии на
живом рынке без риска.

## systemd (автоподнятие)
```bash
sudo tee /etc/systemd/system/pseudo-opt-bot.service <<'UNIT'
[Unit]
Description=Pseudo-Options Scalper
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/pseudo-opt-bot
ExecStart=/opt/pseudo-opt-bot/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now pseudo-opt-bot
journalctl -u pseudo-opt-bot -f
```
Замени `/opt/pseudo-opt-bot` на фактический путь.

## Порядок обкатки
1. **Testnet** (`TESTNET=true`, ключи с https://testnet.bybit.com) — 1–2 недели, WR и PF в `bot.log`.
2. Только после стабильного плюса — `TESTNET=false` и **минимальный** реальный депозит.
3. VPS ближе к серверам Bybit (Сингапур/Токио) — postOnly долетают за 10–20 мс.
4. API-ключи: только «Чтение + Контракты», вывод средств запрещён, IP-whitelist VPS.

## Замечания честности
- 1m-скальпинг чувствителен к задержкам и проскальзыванию на закрытии стопов. Дневной стоп-аут (−4%) защищает от чёрных дней.
- На Cloudflare Workers бота размещать нельзя (нет постоянного процесса / WebSocket, лимиты CPU) — бот живёт на VPS/компьютере. Cloudflare пригоден только для статичной панели мониторинга.
- Телеграм-уведомления опциональны: `TG_TOKEN` и `TG_CHAT_ID`.
