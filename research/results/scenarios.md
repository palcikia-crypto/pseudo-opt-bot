# Сценарный анализ (scenarios.py): сырой эдж vs комиссии

Вариантов (символ×mask×idea×exit) с raw-IS+ и n>=25: 326. Из них raw-OOS+: 49. Выживают при maker-maker 4bp или vip 5bp: 5.

| symbol | mask | idea | exit | IS n/sumR(raw) | OOS n | OOS raw | OOS full | OOS mm4 | OOS vip |
|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | vol1.5 | pullback | trail:2.0 | 2114/31.8 | 1237 | 73.3 | -291.1 | -68.1 | -188.5 |
| BTCUSDT | loose | pullback | trail:2.0 | 2700/48.6 | 1504 | 69.2 | -393.4 | -108.5 | -263.8 |
| SOLUSDT | loose | mr_vwap | trail:2.0 | 1552/2.4 | 805 | 56.2 | -140.2 | -22.0 | -84.2 |
| SOLUSDT | loose | mr_vwap | wide:3.5x3.0 | 1361/31.3 | 712 | 52.8 | -63.8 | 7.6 | -30.9 |
| BTCUSDT | vol1.8 | pullback | trail:2.0 | 1187/35.3 | 644 | 45.3 | -105.2 | -13.7 | -62.6 |
| SOLUSDT | vol1.5 | mr_vwap | wide:3.5x3.0 | 1256/42.7 | 687 | 44.1 | -61.9 | 3.2 | -32.1 |
| XRPUSDT | loose | donchian | rr:1.5x2.5 | 1331/29.1 | 745 | 34.8 | -170.2 | -48.3 | -111.2 |
| ARBUSDT | vol1.5 | donchian | rr:1.5x2.5 | 1298/2.3 | 660 | 32.5 | -96.8 | -20.7 | -59.3 |
| XRPUSDT | loose | trend_bb1h | rr:1.5x2.5 | 1507/34.2 | 789 | 27.4 | -186.2 | -59.3 | -124.7 |
| XRPUSDT | loose | donchian | time:45m | 1161/49.9 | 645 | 24.6 | -141.0 | -48.2 | -91.3 |
| BTCUSDT | vol1.8 | pullback | rr:1.5x2.5 | 1497/63.1 | 746 | 22.8 | -192.4 | -63.0 | -131.0 |
| SOLUSDT | vol1.8 | mr_vwap | wide:3.5x3.0 | 872/54.9 | 513 | 22.2 | -42.1 | -2.4 | -24.1 |
| BTCUSDT | loose | pullback | rr:1.5x2.5 | 3068/25.0 | 1639 | 18.7 | -505.9 | -187.8 | -357.2 |
| BTCUSDT | vol1.8 | pullback | wide:3.5x3.0 | 902/53.8 | 493 | 18.7 | -49.9 | -7.3 | -30.8 |
| XRPUSDT | vol1.5 | donchian | time:45m | 948/31.0 | 552 | 18.7 | -117.5 | -41.5 | -76.5 |
| BTCUSDT | vol1.8 | mr_rsi5 | rr:1.5x2.5 | 713/46.2 | 378 | 17.4 | -88.3 | -25.5 | -57.9 |
| BNBUSDT | vol1.8 | squeeze | wide:3.5x3.0 | 362/11.3 | 155 | 16.9 | -4.3 | 8.5 | 1.8 |
| BNBUSDT | vol1.8 | trend_bb15 | wide:3.5x3.0 | 383/12.7 | 165 | 15.3 | -7.1 | 6.4 | -0.7 |
| XRPUSDT | loose | donchian | wide:3.5x3.0 | 1062/64.7 | 640 | 14.8 | -85.3 | -24.6 | -56.9 |
| BTCUSDT | vol1.5 | mr_rsi5 | rr:1.5x2.5 | 1233/13.6 | 681 | 13.9 | -197.1 | -70.2 | -136.9 |

## Выводы

- Сырой (ZERO-fee) эдж существует у 49 вариантов, лидер: **BTCUSDT/pullback/trail:2.0** (OOS raw 73.3R на 1237 сделках).

- При практике maker-maker (лимитный вход+выход) выживают 5 вариантов — это путь: постOnly входы + лимитные TP/BE (уже в боте) есть, проверять на paper.

