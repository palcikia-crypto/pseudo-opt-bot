"""research/fees_real.py — измерение РЕАЛЬНЫХ издержек с биржи (public API):
- спред (best bid/ask из стакана),
- проскальзывание рыночного выхода для нашего типичного объёма
  (VWAP-проход по стакану, 50 уровней глубины),
Результат: research/results/fees_real.json
"""
import json
import os
import sys
import time
import urllib.request

BYBIT = 'https://api.bybit.com/v5/market/{ep}'
MAKER = 0.02
TAKER = 0.055


def _get(ep, params):
    q = '&'.join('%s=%s' % (str(k), str(v)) for k, v in params.items())
    url = BYBIT.format(ep=ep) + '?' + q
    for a in range(3):
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                d = json.load(r)
            if d.get('retCode') == 0:
                return d['result']
        except Exception:
            time.sleep(1.5 * (a + 1))
    return None


def _vwap_cross(levels, qty):
    need = qty
    cost, got = 0.0, 0.0
    for px, size in levels:
        take = min(size, need)
        cost += take * px
        got += take
        need -= take
        if need <= 1e-12:
            return cost / got
    return float('inf')


def measure(symbol, notional_usdt=5000.0):
    r = _get('orderbook', dict(category='linear', symbol=symbol, limit=50))
    if not r:
        return None
    bids = [(float(p), float(s)) for p, s in r.get('b', [])]
    asks = [(float(p), float(s)) for p, s in r.get('a', [])]
    if not bids or not asks:
        return None
    best_bid, best_ask = bids[0][0], asks[0][0]
    mid = (best_bid + best_ask) / 2.0
    spread_bps = (best_ask - best_bid) / mid * 10000.0
    qty_needed = notional_usdt / mid
    vwap = _vwap_cross(bids, qty_needed)
    impact_bps = (best_bid - vwap) / mid * 10000.0 if vwap != float('inf') else None
    depth20 = sum(s for _, s in bids[:20]) * mid
    return dict(symbol=symbol, spread_bps=round(spread_bps, 2),
                impact_exit_bps=round(impact_bps, 2) if impact_bps is not None else None,
                depth20_usdt=round(depth20, 0))


def main():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import SYMBOLS
    out = dict(ts=int(time.time() * 1000), notional_usdt=5000,
               maker_fee_pct=MAKER, taker_fee_pct=TAKER, rows=[])
    for s in SYMBOLS:
        m = measure(s.replace('/', '').split(':')[0])
        if m:
            out['rows'].append(m)
            print('%-14s spread=%s bps  impact5k=%s bps  depth20=$%s' % (
                m['symbol'], m['spread_bps'], m['impact_exit_bps'], m['depth20_usdt']))
        time.sleep(0.25)
    os.makedirs('research/results', exist_ok=True)
    with open('research/results/fees_real.json', 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print('OK: research/results/fees_real.json')


if __name__ == '__main__':
    main()
