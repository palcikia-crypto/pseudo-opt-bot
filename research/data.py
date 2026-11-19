"""research/data.py — загрузка и кэш исторических 1m-свечей Bybit (public REST)."""
import csv
import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.request

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
BYBIT_HOSTS = ('api.bybit.com', 'api.bytick.com')
BYBIT_KLINE = 'https://{host}/v5/market/kline?category=linear&symbol={s}&interval=1&limit=1000&end={end_ms}'
MAX_RETRY = 6


def cache_path(symbol):
    return os.path.join(CACHE_DIR, symbol.replace('/', '_').replace(':', '_') + '_1m.csv.gz')


def _parse(rows):
    rows = sorted(rows, key=lambda r: int(r[0]))
    return [(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]))
            for r in rows]


def _fetch_page(symbol, end_ms):
    """Одна страница до 1000 баров с ретраями и ротацией хостов."""
    for attempt in range(MAX_RETRY):
        host = BYBIT_HOSTS[attempt % len(BYBIT_HOSTS)]
        url = BYBIT_KLINE.format(host=host,
                                 s=symbol.replace('/', '').split(':')[0], end_ms=end_ms)
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                d = json.load(resp)
            if d.get('retCode') == 0:
                return d['result'].get('list') or []
            if d.get('retCode') in (10016, 10006):
                time.sleep(1.5 * (attempt + 1))
                continue
            raise RuntimeError('%s: retCode=%s %s' % (
                symbol, d.get('retCode'), d.get('retMsg')))
        except (urllib.error.URLError, OSError):
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError('%s: kline @end=%s — не удалось' % (symbol, end_ms))


def _fetch_pages(symbol, end_ms, bars_needed):
    from concurrent.futures import ThreadPoolExecutor
    n_pages = (bars_needed + 999) // 1000
    ends = [end_ms - p * 1000 * 60000 for p in range(n_pages)]
    out = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for rows in pool.map(lambda e: _fetch_page(symbol, e), ends):
            out.extend(rows)
    if not out:
        return []
    seen, dedup = set(), []
    for r in sorted(out, key=lambda x: int(x[0])):
        if r[0] not in seen:
            seen.add(r[0])
            dedup.append(r)
    return _parse(dedup)


def load_1m(symbol, days=7, refresh=False):
    """[(ts_ms, o, h, l, c, v)] oldest-first, из кэша или с биржи."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = cache_path(symbol)
    if refresh or not os.path.exists(path):
        end_ms = int(time.time() * 1000)
        bars = _fetch_pages(symbol, end_ms, days * 1440 + 50)
        tmp = path + '.tmp'
        with gzip.open(tmp, 'wt', newline='') as f:
            w = csv.writer(f)
            w.writerows(bars)
        os.replace(tmp, path)
        return bars
    with gzip.open(path, 'rt') as f:
        return [tuple([int(r[0])] + [float(x) for x in r[1:6]])
                for r in csv.reader(f) if r]


def to_5m(bars_1m):
    """Агрегация 1m -> 5m по границам 5-минутных epoch-бакетов (как у Bybit)."""
    out = []
    bucket = None
    for ts, o, h, l, c, v in bars_1m:
        b = ts // 300000
        if bucket is None or bucket[0] != b:
            if bucket is not None:
                out.append(bucket[1])
            bucket = (b, [b * 300000, o, h, l, c, v])
        else:
            r = bucket[1]
            r[2] = max(r[2], h)
            r[3] = min(r[3], l)
            r[4] = c
            r[5] += v
    if bucket is not None:
        out.append(bucket[1])
    return out


def main():
    symbols = [s.strip() for s in
               (os.getenv('SYMBOLS') or
                'BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT,XRP/USDT:USDT,DOGE/USDT:USDT,'
                'SUI/USDT:USDT,AVAX/USDT:USDT,LINK/USDT:USDT,ADA/USDT:USDT,BNB/USDT:USDT,'
                'NEAR/USDT:USDT,APT/USDT:USDT,ARB/USDT:USDT,OP/USDT:USDT,SEI/USDT:USDT').split(',')]
    days = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.getenv('RESEARCH_DAYS', '7'))
    refresh = '--refresh' in sys.argv
    for s in symbols:
        bars = load_1m(s, days=days, refresh=refresh)
        print('%-16s %6d баров 1m (%.1f дн)' % (s, len(bars), len(bars) / 1440.0))


if __name__ == '__main__':
    main()
