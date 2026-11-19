"""research/funding.py — история funding rate Bybit (public) по всем символам."""
import json
import os
import sys
import time
import urllib.request

funding_ep = 'https://api.bybit.com/v5/market/funding/history?category=linear&symbol={s}&limit=200'


def main():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import SYMBOLS
    out = {}
    for s in SYMBOLS:
        sym = s.replace('/', '').split(':')[0]
        try:
            with urllib.request.urlopen(funding_ep.format(s=sym), timeout=15) as r:
                d = json.load(r)
            rows = d.get('result', {}).get('list') or []
            rates = [float(x['fundingRate']) for x in rows]
            out[sym] = dict(n=len(rates),
                            mean=round(sum(rates) / len(rates), 6) if rates else None,
                            std=round((sum((x - sum(rates) / len(rates)) ** 2 for x in rates)
                                       / len(rates)) ** 0.5, 6) if rates else None)
        except Exception as e:
            out[sym] = dict(error=str(e))
        time.sleep(0.2)
    os.makedirs('research/results', exist_ok=True)
    with open('research/results/funding.json', 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
