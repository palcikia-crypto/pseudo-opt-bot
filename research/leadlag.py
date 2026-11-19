"""research/leadlag.py — «гонка за битком»: измеряем, как движение BTC
передаётся альтам во времени (lead-lag).

Для каждой монеты: кросс-корреляция 1m-доходностей альт(t) vs BTC(t-k),
k = 0..30 минут. Отчёт: лучший лаг (сколько минут альт «догоняет» BTC),
корреляция, доля объяснённой дисперсии (R²), «скорость передачи» (beta).

Запуск: ./venv/bin/python research/leadlag.py
"""
import datetime
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.data import load_1m

LAGS = (0, 1, 2, 3, 5, 8, 12, 20, 30)


def main():
    from config import SYMBOLS
    full = [s.replace('/', '').split(':')[0] for s in SYMBOLS]
    rets = {}
    for sym in full:
        b = load_1m(sym)
        c = np.array([x[4] for x in b], dtype=float)
        rets[sym] = np.diff(c) / c[:-1]
    n = min(len(r) for r in rets.values())
    btc = rets['BTCUSDT'][-n:]
    out = {}
    print('lag = на сколько минут альт отстаёт от BTC')
    for sym in full:
        if sym == 'BTCUSDT':
            continue
        ra = rets[sym][-n:]
        row, best = {}, (0, 0.0)
        for k in LAGS:
            a = ra[k:]
            b = btc[:btc.size - k] if k else btc
            if a.size < 100:
                continue
            c = float(np.corrcoef(a, b)[0, 1])
            row[k] = round(c, 4)
            if k > 0 and abs(c) > abs(best[1]):
                best = (k, round(c, 4))
        k = best[0] if best[0] > 0 else 1
        a, b = ra[k:], btc[:btc.size - k]
        beta = float(np.dot(a, b) / max(np.dot(b, b), 1e-18))
        out[sym] = dict(best_lag_min=k, best_corr=best[1], r2=round(best[1] ** 2, 4),
                        beta_btc=round(beta, 3), corr_by_lag=row)
        print('%-8s lag=%2dмин corr=%+.3f R2=%.3f beta=%.2f' % (
            sym.replace('USDT', ''), k, best[1], best[1] ** 2, beta), flush=True)

    os.makedirs('research/results', exist_ok=True)
    json.dump(dict(measured=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   lags=LAGS, coins=out),
              open('research/results/leadlag.json', 'w'), indent=1, ensure_ascii=False)
    print('saved research/results/leadlag.json')


if __name__ == '__main__':
    main()
