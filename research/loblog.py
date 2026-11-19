"""research/loblog.py — живой логгер микроструктуры Bybit (стакан + ленты сделок).

Пишет каждые ~10с: bid/ask, спред, дисбаланс первых 25 уровней, top-25 VWAP,
последние сделки (агрессивный buy/sell дельта). Результат — датасет для будущих
бэктестов микро-сигналов (исторического стакана у Bybit нет — собираем сами).

Запуск (демон): setsid nohup ./venv/bin/python research/loblog.py > research/loblog.log 2>&1 &
"""
import csv
import datetime as dt
import gzip
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SYMBOLS = os.getenv('LOB_SYMBOLS', 'BTCUSDT,SOLUSDT,BNBUSDT,XRPUSDT').split(',')
INTERVAL = float(os.getenv('LOB_INTERVAL', '10'))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')

_ep_ob = 'https://api.bybit.com/v5/market/orderbook?category=linear&symbol={s}&limit=50'
_ep_tr = 'https://api.bybit.com/v5/market/recent-trade?category=linear&symbol={s}&limit=60'


def get(url):
    for a in range(4):
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                return json.load(r)
        except Exception:
            time.sleep(1 + a)
    return None


def main():
    os.makedirs(OUT, exist_ok=True)
    files, wrs = {}, {}
    day = None
    while True:
        try:
            now = dt.datetime.now(dt.timezone.utc)
            if now.date() != day:
                for f in files.values():
                    f.close()
                files, wrs = {}, {}
                day = now.date()
            ts = now.strftime('%Y-%m-%dT%H:%M:%SZ')
            for sym in SYMBOLS:
                ob = get(_ep_ob.format(s=sym))
                tr = get(_ep_tr.format(s=sym))
                if not ob or ob.get('retCode') != 0 or not tr:
                    continue
                bids = [(float(b[0]), float(b[1])) for b in ob['result'].get('b', [])]
                asks = [(float(a[0]), float(a[1])) for a in ob['result'].get('a', [])]
                if not bids or not asks:
                    continue
                bv25 = sum(p * q for p, q in bids[:25])
                av25 = sum(p * q for p, q in asks[:25])
                imb = (bv25 - av25) / (bv25 + av25)
                spread_bps = (asks[0][0] - bids[0][0]) / bids[0][0] * 1e4
                lbid = sum(p * q * p for p, q in bids[:25]) / bv25
                lask = sum(p * q * p for p, q in asks[:25]) / av25
                tlist = tr.get('result', {}).get('list', [])
                d_buy = d_sell = 0.0
                last_ts = 0
                for t in tlist:
                    tms = int(t.get('tradeTimeMs') or 0)
                    if tms <= last_ts:
                        continue
                    last_ts = max(last_ts, tms)
                    sz = float(t.get('size', 0))
                    if t.get('side') == 'Buy':
                        d_buy += sz
                    else:
                        d_sell += sz
                if sym not in files:
                    path = os.path.join(OUT, 'lob_%s_%s.csv.gz' % (sym, now.strftime('%Y%m%d')))
                    files[sym] = gzip.open(path, 'at', newline='')
                    wrs[sym] = csv.writer(files[sym])
                    if files[sym].tell() == 0:
                        wrs[sym].writerow(['ts', 'bid', 'ask', 'spread_bps', 'imb25',
                                           'lbid', 'lask', 'd_buy', 'd_sell'])
                wrs[sym].writerow([ts, bids[0][0], asks[0][0], round(spread_bps, 3),
                                   round(imb, 4), lbid, lask, round(d_buy, 4),
                                   round(d_sell, 4)])
            for f in files.values():
                f.flush()
        except Exception as e:
            print('[loblog] %s' % e, flush=True)
        time.sleep(INTERVAL)


if __name__ == '__main__':
    main()
