"""research/edge_scan.py — поиск закономерностей в результатах сделок:
1) логистическая регрессия (numpy, без зависимостей): какие признаки при входе
   предсказывают исход (win/loss); time-series CV (3 фолда без перемешивания);
2) бакет-анализ: винрейт/средний R по часам суток, ATR-, объёмным корзинам.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FEATS = ['atr_pct', 'rsi', 'vol_ratio', 'bb_over', 'brk_over', 'trend5']


def feats_matrix(trades):
    X, y, aux = [], [], []
    for t in trades:
        f = t['feats']
        h = f['hour']
        X.append([f['atr_pct'], f['rsi'] - 50.0, f['vol_ratio'],
                  f['bb_over'], f['brk_over'], f['trend5'],
                  math.sin(2 * math.pi * h / 24.0), math.cos(2 * math.pi * h / 24.0),
                  1.0 if f['side'] == 'long' else -1.0])
        y.append(1.0 if t['pnl'] > 0 else 0.0)
        aux.append(dict(ts=t['ts_entry'], R=t['R'], hour=h,
                        atr=f['atr_pct'], vol=f['vol_ratio']))
    return np.array(X, dtype=float), np.array(y), aux


def _fit(X, y, l2=1e-3, iters=600, lr=0.08):
    m, k = X.shape
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    Z = (X - mu) / sd
    Z = np.hstack([Z, np.ones((m, 1))])
    w = np.zeros(k + 1)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Z @ w))
        g = Z.T @ (p - y) / m + l2 * w
        g[-1] -= l2 * w[-1]
        w -= lr * np.clip(g, -5.0, 5.0)
    return w, mu, sd


def _predict(w, mu, sd, X):
    Z = (X - mu) / sd
    Z = np.hstack([Z, np.ones((len(X), 1))])
    return 1.0 / (1.0 + np.exp(-Z @ w))


def _auc(y, p):
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return 0.5
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def cv_auc(trades, k=3):
    X, y, aux = feats_matrix(trades)
    ts = np.array([a['ts'] for a in aux])
    idx = np.argsort(ts)
    splits = np.array_split(idx, k)
    stats = []
    for i in range(k):
        te = splits[i]
        tr = np.concatenate([splits[j] for j in range(k) if j != i])
        w, mu, sd = _fit(X[tr], y[tr])
        p = _predict(w, mu, sd, X[te])
        hi = p > 0.5
        base_wr = float(y[te].mean())
        hi_wr = float(y[te][hi].mean()) if hi.any() else float('nan')
        hi_R = float(np.mean([aux[j]['R'] for j in te[hi]])) if hi.any() else 0.0
        all_R = float(np.mean([aux[j]['R'] for j in te]))
        stats.append(dict(fold=i, n_test=int(te.size), auc=round(_auc(y[te], p), 3),
                          acc=round(float(((p > 0.5) == (y[te] > 0.5)).mean()), 3),
                          wr_base=round(base_wr, 3),
                          wr_p05=None if not hi.any() else round(hi_wr, 3),
                          R_base=round(all_R, 3), R_p05=round(hi_R, 3),
                          n_p05=int(hi.sum())))
    return stats


def bucket_stats(trades, keyfn):
    buckets = {}
    for t in trades:
        b = keyfn(t)
        d = buckets.setdefault(b, dict(n=0, wins=0, R=0.0))
        d['n'] += 1
        d['wins'] += 1 if t['pnl'] > 0 else 0
        d['R'] += t['R']
    out = []
    for b, d in sorted(buckets.items()):
        out.append(dict(bucket=b, n=d['n'],
                        win_rate=round(100.0 * d['wins'] / d['n'], 1),
                        avg_R=round(d['R'] / d['n'], 3), sum_R=round(d['R'], 2)))
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'research/results/base.json'
    with open(path) as f:
        trades = json.load(f)['trades']
    if len(trades) < 30:
        print('слишком мало сделок (%d) для скана' % len(trades))
        return
    out = dict(
        n_trades=len(trades),
        cv=cv_auc(trades),
        by_hour=bucket_stats(trades, lambda t: t['feats']['hour']),
        by_atr=bucket_stats(trades, lambda t: (
            'atr_<0.10' if t['feats']['atr_pct'] < 0.10 else
            'atr_0.10-0.20' if t['feats']['atr_pct'] < 0.20 else
            'atr_0.20-0.35' if t['feats']['atr_pct'] < 0.35 else 'atr_>=0.35')),
        by_vol=bucket_stats(trades, lambda t: (
            'vol_<2' if t['feats']['vol_ratio'] < 2.0 else
            'vol_2-4' if t['feats']['vol_ratio'] < 4.0 else 'vol_>=4')),
        by_side=bucket_stats(trades, lambda t: t['feats']['side']),
        by_symbol=bucket_stats(trades, lambda t: t['symbol'].split('/')[0]),
    )
    os.makedirs('research/results', exist_ok=True)
    with open('research/results/edge_scan.json', 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps(dict(cv=out['cv'], by_side=out['by_side'],
                          by_atr=out['by_atr'], by_vol=out['by_vol']),
                     indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
