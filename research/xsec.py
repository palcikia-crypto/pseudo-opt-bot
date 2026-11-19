"""research/xsec.py — кросс-секционный моментум/реверс по 15 альтам.

На 15-минутной сетке: ранжируем монеты по доходности за окно K, лонг top-3 /
шорт bottom-3, держим H минут, равновзвешенный L/S-спред. Комиссии: 2 ноги x
(5.5bp тейкер in + 5.5bp out) = 22bp за спред-период. IS/OOS 120/60 дней.

Запуск: ./venv/bin/python research/xsec.py
"""
import datetime
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.data import load_1m

DAY = 24 * 3600 * 1000
MOM_WINDOWS = (15, 60, 180, 360, 1440)
HOLDS = (15, 60, 180)
TOPN = 3
STEP = 15
FEE_PER_SPREAD = 2 * (0.00055 + 0.00055)
RISK_ANN = 365 * 24 * 4 * 60 / 15


def load_grid():
    from config import SYMBOLS
    full = [s.replace('/', '').split(':')[0] for s in SYMBOLS]
    raw = {sym: load_1m(sym) for sym in full}
    ref = raw['BTCUSDT']
    ref_ts = np.array([b[0] for b in ref], dtype=np.int64)
    px = {}
    for sym, bars in raw.items():
        ts = np.array([b[0] for b in bars], dtype=np.int64)
        c = np.array([b[4] for b in bars], dtype=float)
        pos = np.searchsorted(ts, ref_ts)
        pv = np.minimum(pos, ts.size - 1)
        ok = (pos < ts.size) & (ts[pv] == ref_ts)
        v = np.where(ok, c[pv], np.nan)
        if np.isfinite(v).mean() > 0.98:
            z = np.where(np.isfinite(v), v, np.nan)
            idx = np.arange(v.size)
            z = np.interp(idx, idx[np.isfinite(v)], v[np.isfinite(v)])
            out = np.where(ok, z, np.nan)
            px[sym] = np.where(ok, out, np.nan)
        else:
            px[sym] = v
    return full, ref_ts, px


def spread_series(mom_w, hold, ref_ts, px, t0, t1):
    syms = [s for s in px if s and not np.all(np.isnan(px[s]))]
    grid = np.arange(t0, t1, STEP * 60000)
    pos_idx = np.searchsorted(ref_ts, grid)
    ok_grid = (pos_idx < ref_ts.size) & (ref_ts[np.minimum(pos_idx, ref_ts.size - 1)] == grid)
    spread, gidx = [], []
    w = mom_w
    for gi, i_lat in enumerate(pos_idx):
        if not ok_grid[gi] or i_lat < w + hold:
            continue
        col = np.array([px[s][i_lat] for s in syms])
        past = np.array([px[s][i_lat - w] for s in syms])
        mom = col / past - 1.0
        fin = np.isfinite(mom)
        if fin.sum() < len(syms) * 0.8:
            continue
        order = np.argsort(np.where(fin, mom, np.nan), kind='stable')
        n = fin.sum()
        if n < 2 * TOPN:
            continue
        top = order[-TOPN:]
        bot = order[:TOPN]
        f = np.array([px[s][min(i_lat + hold, ref_ts.size - 1)] for s in syms])
        ret_l = np.nanmean(f[top] / col[top] - 1.0)
        ret_s = np.nanmean(f[bot] / col[bot] - 1.0)
        sp = ret_l - ret_s - FEE_PER_SPREAD
        if np.isfinite(sp):
            spread.append(sp)
            gidx.append(grid[gi])
    return np.array(gidx), np.array(spread)


def summarize(gidx, sp, lo, hi, label):
    m = (gidx >= lo) & (gidx < hi)
    s = sp[m]
    if s.size < 50:
        return dict(label=label, n=0)
    t = float(s.mean() / (s.std() / np.sqrt(s.size))) if s.std() > 0 else 0.0
    return dict(label=label, n=int(s.size), mean_bp=round(float(s.mean()) * 1e4, 2),
                pos_pct=round(100.0 * float((s > 0).mean()), 1), tstat=round(t, 2),
                sum_bp=round(float(s.sum()) * 1e4, 1))


def main():
    full, ref_ts, px = load_grid()
    t0 = int(ref_ts[0]); t1 = int(ref_ts[-1])
    is_end, oos_start = t0 + 120 * DAY, t1 - 60 * DAY
    names = {s: s.replace('USDT', '') for s in full}
    print('slot: %s .. %s | IS до %s, OOS с %s' % (
        datetime.datetime.fromtimestamp(t0 / 1000, datetime.timezone.utc).strftime('%Y-%m-%d'),
        datetime.datetime.fromtimestamp(t1 / 1000, datetime.timezone.utc).strftime('%Y-%m-%d'),
        datetime.datetime.fromtimestamp(is_end / 1000, datetime.timezone.utc).strftime('%Y-%m-%d'),
        datetime.datetime.fromtimestamp(oos_start / 1000, datetime.timezone.utc).strftime('%Y-%m-%d')),
        flush=True)
    results = []
    for w in MOM_WINDOWS:
        for h in HOLDS:
            gidx, sp = spread_series(w, h, ref_ts, px, t0, t1)
            row = dict(mom_min=w, hold_min=h,
                       IS=summarize(gidx, sp, t0, is_end, 'IS'),
                       OOS=summarize(gidx, sp, oos_start, t1, 'OOS'),
                       FULL=summarize(gidx, sp, t0, t1, 'FULL'))
            results.append(row)
            print('mom=%4dm hold=%3dm IS %s | OOS %s' % (
                w, h, row['IS'], row['OOS']), flush=True)

    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    os.makedirs('research/results', exist_ok=True)
    json.dump(dict(measured=stamp, topn=TOPN, step_min=STEP, fee_spread_bp=22.0,
                   results=results),
              open('research/results/xsec.json', 'w'), indent=1, ensure_ascii=False)
    best = [r for r in results
            if r['OOS'].get('n', 0) >= 50 and r['OOS'].get('mean_bp', 0) > 0
            and r['OOS'].get('tstat', 0) >= 2.0]
    lines = ['# Кросс-секционный моментум/реверс (топ-3 / анти-топ-3, спред, fee 22bp)', '',
             '| mom | hold | IS n | IS bp | IS t | OOS n | OOS bp | OOS t | OOS %>0 |',
             '|-----|------|-------|-------|------|-------|--------|-------|---------|']
    for r in results:
        i, o = r['IS'], r['OOS']
        lines.append('| %dm | %dm | %s | %+.2f | %s | %s | %+.2f | %s | %s |' % (
            r['mom_min'], r['hold_min'], i.get('n', 0), i.get('mean_bp', 0.0),
            i.get('tstat', '-') if 'tstat' in i else '-',
            o.get('n', 0), o.get('mean_bp', 0.0), o.get('tstat', '-'),
            o.get('pos_pct', '-')))
    lines += ['', '**OOS-выжившие (t>=2, n>=50, bp>0)**: %s' % (
        ', '.join('%dm/%dm' % (r['mom_min'], r['hold_min']) for r in best) or 'нет')]
    open('research/results/xsec.md', 'w').write('\n'.join(lines) + '\n')
    print('saved research/results/xsec.json + .md; OOS-выживших: %d' % len(best),
          flush=True)


if __name__ == '__main__':
    main()
