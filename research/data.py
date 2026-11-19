"""research/data.py — загрузка и кэш исторических 1m-свечей Bybit (public REST)."""
import csv
import gzip
import json
import os
import sys
import time
import urllib.request

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
BYBIT_KLINE = 'https://api.bybit.com/v5/market/kline?category=linear&symbol={s}&interval=1&limit=1000&end={end_ms}'
MAX_RETRY = 4


def cache_path(symbol):
    return os.path.join(CACHE_DIR, symbol.replace('/', '_').replace(':', '_') + '_1m.csv.gz')


def _parse(rows):
    rows = sorted(rows, key=lambda r: int(r[0]))
    return [(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]))
            for r in rows]


def _fetch(symbol, end_ms, bars_needed):
    got, out, end = 0, [], end_ms
    while got < bars_needed:
        url = BYBIT_KLINE.format(s=symbol.replace('/', '').split(':')[0], end_ms=end)
        for attempt in range(MAX_RETRY):
            try:
                with urllib.request.urlopen(url, timeout=15) as resp:
                    data = json.load(resp)
                break
            except Exception as e:
                if attempt == MAX_RETRY - 1:
                    raise
                time.sleep(2 * (attempt + 1))
        if data.get('retCode') != 0:
            raise RuntimeError('%s: retCode=%s %s' % (symbol, data.get('retCode'), data.get('retMsg')))
        rows = data['result'].get('list') or []
        if not rows:
            break
        parsed = _parse(rows)
        out = parsed + out
        got += len(parsed)
        end = parsed[0][0] - 1
        time.sleep(0.15)
    if out:
        seen, dedup = set(), []
        for r in out:
            if r[0] not in seen:
                seen.add(r[0])
                dedup.append(r)
        out = dedup
    return out


def load_1m(symbol, days=7, refresh=False):
    """[(ts_ms, o, h, l, c, v)] oldest-first, из кэша или с биржи."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = cache_path(symbol)
    if refresh or not os.path.exists(path):
        end_ms = int(time.time() * 1000)
        bars = _fetch(symbol, end_ms, days * 1440 + 50)
        tmp = path + '.tmp'
        with gzip.open(tmp, 'wt', newline='') as f:
            w = csv.writer(f)
            w.writerows(bars)
        os.replace(tmp, path)
        return bars
    with gzip.open(path, 'rt') as f:
        return [tuple([int(r[0])] + [float(x) for x in r[1:]])
                for r in csv.reader(f)]


def to_5m(bars_1m):
    """Агрегация 1m -> 5m по границам epoch-минут (совпадает со.Bybit)."""
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
