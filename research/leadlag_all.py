"""research/leadlag_all.py — «кто за кем идёт»: все-парный lead-lag по 15 монетам + событийное
исследование BTC-шоков + вола-спилловер.

Для каждой упорядоченной пары (L→F): corr(возврат F(t), L(t-k)), k = 1..30 мин —
находим лучший лаг и корреляцию. Повторяем в обратную сторону (F→L): определяем,
кто лидер, кто последователь. BTC-шоки (|1m-ret| > q99.5): сдаёт ли альт следующую
минуту в ту же сторону и насколько. Вола-спилловер: corr(вола BTC, вола альт +10м).

Запуск: ./venv/bin/python research/leadlag_all.py
"""
import datetime
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.data import load_1m

LAGS = (1, 2, 3, 5, 10, 15, 30)
VOL_WINDOW = 30
VOL_SPILL_LAG = 10


def load_aligned():
    from config import SYMBOLS
    full = [s.replace('/', '').split(':')[0] for s in SYMBOLS]
    raw = {sym: load_1m(sym) for sym in full}
    ref_ts = np.array([b[0] for b in raw['BTCUSDT']], dtype=np.int64)
    rets = {}
    for sym, bars in raw.items():
        ts = np.array([b[0] for b in bars], dtype=np.int64)
        c = np.array([b[4] for b in bars], dtype=float)
        pos = np.searchsorted(ts, ref_ts)
        pv = np.minimum(pos, ts.size - 1)
        valid = (pos < ts.size) & (ts[pv] == ref_ts)
        r = np.where(valid, np.nan, 0.0)
        r_full = np.full(c.size, np.nan)
        r_full[1:] = np.diff(c) / c[:-1]
        r[valid] = r_full[pv[valid]]
        rets[sym] = r
    return full, rets


def _corr_nan(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5000:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def pair_matrix(full, rets):
    rows = []
    for a, b in itertools.combinations(full, 2):
        best = None
        for lead, fol in ((a, b), (b, a)):
            for k in LAGS:
                c = _corr_nan(rets[fol][k:], rets[lead][:rets[lead].size - k])
                if not np.isnan(c) and (best is None or abs(c) > abs(best[2])):
                    best = (lead, fol, round(c, 4), k)
        if best:
            rows.append(dict(lead=best[0], fol=best[1], corr=best[2], lag_min=best[3]))
    return rows


def btc_shock_study(full, rets, btc_sym='BTCUSDT'):
    rb = rets[btc_sym]
    fin = np.nan_to_num(np.abs(rb), nan=0.0)
    thr = float(np.quantile(fin[fin > 0], 0.995))
    ev = np.flatnonzero(np.isfinite(rb) & (np.abs(rb) > thr))
    out = {}
    for sym in full:
        if sym == btc_sym:
            continue
        ra = rets[sym]
        nxt = np.stack([np.where(np.isfinite(ra[ev + j]), ra[ev + j], np.nan)
                        for j in range(1, 6)])
        mean_next = float(np.nanmean(nxt))
        agree = float(np.nanmean(np.sign(nxt) == np.sign(rb[ev]))) * 100.0
        base_agree = 50.0
        out[sym] = dict(threshold_pct=round(thr * 100, 3), n_events=int(ev.size),
                        mean_next1m_5m_bp=round(mean_next * 1e4, 2),
                        agree_next_pct=round(agree, 1), base_agree_pct=base_agree)
    return out


def vol_spillover(full, rets, btc_sym='BTCUSDT'):
    def vol(r):
        ar = np.abs(np.nan_to_num(r))
        cs = np.concatenate(([0.0], np.cumsum(ar)))
        idx = np.maximum(np.arange(ar.size) - VOL_WINDOW, 0)
        return cs[1:] - cs[idx]
    vb = vol(rets[btc_sym])
    out = {}
    for sym in full:
        if sym == btc_sym:
            continue
        va = vol(rets[sym])
        c = _corr_nan(va[VOL_SPILL_LAG:], vb[:vb.size - VOL_SPILL_LAG])
        c0 = _corr_nan(va, vb)
        out[sym] = dict(corr_vlag=round(c, 3) if not np.isnan(c) else None,
                        corr_contemp=round(c0, 3) if not np.isnan(c0) else None)
    return out


def main():
    full, rets = load_aligned()
    pairs = pair_matrix(full, rets)
    pairs.sort(key=lambda p: abs(p['corr']), reverse=True)
    shocks = btc_shock_study(full, rets)
    spill = vol_spillover(full, rets)

    print('=== TOP-15 lead-lag (|corr| на мак. лаге) ===', flush=True)
    for p in pairs[:15]:
        print('%-8s -> %-8s lag=%2dм corr=%+.3f' % (
            p['lead'].replace('USDT', ''), p['fol'].replace('USDT', ''),
            p['lag_min'], p['corr']), flush=True)
    print('=== BTC-шоки: реакция альтов 1..5м после (bp) / совпадение знака ===',
          flush=True)
    for sym, d in shocks.items():
        print('%-8s mean=%+6.1fbp agree=%4.1f%% (thr %.3f%%, n=%d)' % (
            sym.replace('USDT', ''), d['mean_next1m_5m_bp'], d['agree_next_pct'],
            d['threshold_pct'], d['n_events']), flush=True)

    os.makedirs('research/results', exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    data = dict(measured=stamp, lags=LAGS, pairs=pairs, btc_shocks=shocks,
                vol_spillover=spill)
    json.dump(data, open('research/results/leadlag_all.json', 'w'),
              indent=1, ensure_ascii=False)

    lines = ['# Lead-lag все-пары + BTC-шоки (%s UTC)' % stamp[:10], '',
             '| № | Лидер | Последователь | Лаг, мин | corr |',
             '|---|-------|---------------|----------|------|']
    for i, p in enumerate(pairs, 1):
        lines.append('| %d | %s | %s | %d | %+.3f |' % (
            i, p['lead'].replace('USDT', ''), p['fol'].replace('USDT', ''),
            p['lag_min'], p['corr']))
    lines += ['', '## Реакция альтов на BTC-шок (|ret| > 0.995 квантиль)',
              '', '| Альт | Ср. дох. 1..5м, bp | Совпадение знака, % | n |', '|---|---|---|---|']
    for sym, d in shocks.items():
        lines.append('| %s | %+.1f | %.1f | %d |' % (
            sym.replace('USDT', ''), d['mean_next1m_5m_bp'], d['agree_next_pct'],
            d['n_events']))
    open('research/results/leadlag_all.md', 'w').write('\n'.join(lines) + '\n')
    print('saved research/results/leadlag_all.json + .md', flush=True)


if __name__ == '__main__':
    main()
