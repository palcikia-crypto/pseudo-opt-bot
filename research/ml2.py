"""research/ml2.py — раунд 8: ML с тройным барьером под торговую модель.

Модель: барьеры TP=+2.5R / SL=-1R (sl = 1.5*ATR1m), горизонт 240 1m-баров
(SL-first при неоднозначности — консервативно). Выборка — каждый 5-й 1m-бар
по всем 15 монетам. 20 фич: рет 1/5/15/60м, ATR%, vol_ratio, squeeze,
дистанции VWAP/EMA20, wick/body, rsi5, час (sin/cos), BTC ret1/ATR%, ret1440,
направление свечи, vol×dir.

Обучение: HistGradientBoostingClassifier (long/short — отдельные модели) на
IS 120д с purging на горизонт + embargo 1 день; OOS 60д — AUC + политика
«торгуем верхние 20% train-вероятностей» через lab.sim с полными комиссиями.

Запуск: ./venv/bin/python research/ml2.py
"""
import datetime
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import research.lab as lab
from research.scenarios import net_R, SCENARIOS

DAY = lab.DAY
STEP = 5
BARRIER_BARS = 240
RR = 2.5
EXIT = dict(sl_atr_mult=1.5, rr=RR, exit='rr')

FEATURES = ['ret1', 'ret5', 'ret15', 'ret60', 'atr_pct', 'vol_ratio', 'squeeze',
            'vw_dist', 'ema_dist', 'up_wick', 'low_wick', 'body_ratio', 'rsi5',
            'sin_h', 'cos_h', 'btc_ret1', 'btc_atr', 'ret1440', 'dir_sign',
            'vol_x_dir']


def cummax_rev(x):
    return np.maximum.accumulate(x[::-1])[::-1]


def cummin_rev(x):
    return np.minimum.accumulate(x[::-1])[::-1]


def build_features(ld, btc):
    n = ld.ts.size
    c = ld.c

    def ret(k):
        r = np.full(n, np.nan)
        if n > k:
            r[k:] = c[k:] / c[:-k] - 1.0
        return np.nan_to_num(r)

    rng = np.maximum(ld.h - ld.l, 1e-12)
    vw = np.where(np.isfinite(ld.vwap), ld.vwap, c)
    em = np.where(np.isfinite(ld.ema1m20), ld.ema1m20, c)
    hour = (ld.ts // 3600000) % 24
    pos = np.searchsorted(btc.ts, ld.ts)
    pv = np.minimum(pos, btc.ts.size - 1)
    okb = (pos < btc.ts.size) & (btc.ts[pv] == ld.ts)
    body_dir = np.sign(c - ld.o)
    return np.column_stack([
        ret(1), ret(5), ret(15), ret(60),
        np.nan_to_num(ld.atr_pct),
        np.where(np.isfinite(ld.vol_ratio), ld.vol_ratio, 0.0),
        np.nan_to_num(ld.squeeze),
        c / vw - 1.0, c / em - 1.0,
        (ld.h - np.maximum(ld.o, c)) / rng,
        (np.minimum(ld.o, c) - ld.l) / rng,
        np.abs(c - ld.o) / rng,
        np.nan_to_num(ld.rsi5),
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.where(okb, np.nan_to_num(btc.ret1[pv]), 0.0),
        np.where(okb, np.nan_to_num(btc.atr_pct[pv]), 0.0),
        ret(1440),
        body_dir,
        np.nan_to_num(ld.vol_ratio) * body_dir,
    ]).astype(np.float32)


def barrier_labels(ld, idxs):
    """[(idx, y_long, y_short)]: +1 = TP раньше SL, иначе -1; без барьера — не включён."""
    h, l_, atr, c = ld.h, ld.l, ld.atr, ld.c
    n = ld.ts.size
    sl_d = 1.5 * np.nan_to_num(atr)
    tp_l = c + RR * sl_d
    sl_l = c - sl_d
    tp_s = c - RR * sl_d
    sl_s = c + sl_d
    out = []
    for i in idxs:
        j0 = int(i) + 1
        j1 = min(j0 + BARRIER_BARS, n)
        if j1 - j0 < BARRIER_BARS:
            continue
        hh = cummax_rev(h[j0:j1])
        ll = cummin_rev(l_[j0:j1])
        ht_l = np.flatnonzero(hh >= tp_l[i])
        hs_l = np.flatnonzero(ll <= sl_l[i])
        yl = 1 if (ht_l.size and (not hs_l.size or ht_l[0] < hs_l[0])) else -1
        ht_s = np.flatnonzero(ll <= tp_s[i])
        hs_s = np.flatnonzero(hh >= sl_s[i])
        ys = 1 if (ht_s.size and (not hs_s.size or ht_s[0] < hs_s[0])) else -1
        out.append((int(i), yl, ys))
    return out


def policy_backtest(labs, meta, models, t1):
    mk, tk = list(SCENARIOS.values())[0]
    agg = []
    for sym, md in meta.items():
        ld = labs[sym]
        clf_l, thr_l, keep_l = models['long']
        clf_s, thr_s, keep_s = models['short']
        rows = np.array(md['ite'])
        probs_l = clf_l.predict_proba(ld.F_policy[rows])[:, 1]
        probs_s = clf_s.predict_proba(ld.F_policy[rows])[:, 1]
        busy = -1
        sym_trades = 0
        for j, i_te in enumerate(rows):
            if int(i_te) <= busy:
                continue
            if keep_l[int(i_te)] and probs_l[j] >= thr_l:
                side = 'long'
            elif keep_s[int(i_te)] and probs_s[j] >= thr_s:
                side = 'short'
            else:
                continue
            t = lab.sim(ld, EXIT, int(i_te), side, t1)
            if t is None:
                continue
            busy = int(i_te) + t['bars_held']
            n_r = net_R(t, mk, tk)
            agg.append(dict(symbol=sym, side=side, R=t['R'], net_R=round(n_r, 4),
                            ts_entry=t['ts_entry']))
            sym_trades += 1
        print('[policy %s] %d trades' % (sym, sym_trades), flush=True)
    return agg


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from config import SYMBOLS
    full = [s.replace('/', '').split(':')[0] for s in SYMBOLS]
    raw = {sym: lab.load_1m(sym, days=190) for sym in full}
    labs = {sym: lab.LabData(sym, b) for sym, b in raw.items()}
    btc = labs['BTCUSDT']
    t0 = int(min(l.ts[0] for l in labs.values()))
    t1 = int(max(l.ts[-1] for l in labs.values()))
    is_end, oos_start = t0 + 120 * DAY, t1 - 60 * DAY
    purge_end = is_end - 1 * DAY
    label_after = BARRIER_BARS * 60000

    X_parts, tl_parts, ts_parts = [], [], []
    meta = {}
    for sym in full:
        ld = labs[sym]
        ld.F_policy = build_features(ld, btc)
        n = ld.ts.size
        grid = np.arange(0, max(n - BARRIER_BARS - 1, 0), STEP, dtype=int)
        grid = grid[ld.ts[grid] >= t0]
        lab_tr = barrier_labels(ld, grid[ld.ts[grid] + label_after < purge_end])
        lab_te = barrier_labels(ld, grid[ld.ts[grid] >= oos_start])
        if not lab_tr or not lab_te:
            print('[feat/%s] пропущен' % sym, flush=True)
            continue
        itr = np.array([x[0] for x in lab_tr])
        ite = np.array([x[0] for x in lab_te])
        meta[sym] = dict(itr=itr, ite=ite)
        X_parts.append(ld.F_policy[itr]); X_parts.append(ld.F_policy[ite])
        tl_parts.append(np.array([x[1] for x in lab_tr]))
        tl_parts.append(np.array([x[1] for x in lab_te]))
        ts_parts.append(np.array([x[2] for x in lab_tr]))
        ts_parts.append(np.array([x[2] for x in lab_te]))
        print('[feat/%s] train %d oos %d' % (sym, itr.size, ite.size), flush=True)

    X = np.concatenate(X_parts)
    L = np.concatenate(tl_parts)
    Y_s = np.concatenate(ts_parts)
    n_tr = sum(m['itr'].size for m in meta.values())
    is_mask = np.zeros(X.shape[0], dtype=bool)
    is_mask[:n_tr] = True

    quality, models = {}, {}
    for side, Y in (('long', L), ('short', Y_s)):
        keep = Y != 0
        trm = is_mask & keep
        tem = (~is_mask) & keep
        yb = (Y == 1).astype(int)
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                             max_leaf_nodes=31, random_state=7)
        clf.fit(X[trm], yb[trm])
        auc = roc_auc_score(yb[tem], clf.predict_proba(X[tem])[:, 1])
        p_kept = clf.predict_proba(X[is_mask & keep])[:, 1]
        thr = float(np.quantile(p_kept, 0.80))
        models[side] = (clf, thr, keep)
        quality[side] = dict(auc=round(float(auc), 4), n_train=int(trm.sum()),
                             n_oos=int(tem.sum()),
                             tp_rate_oos=round(float(yb[tem].mean()), 4),
                             prob_threshold=round(thr, 4))
        print('%s: AUC=%.4f train=%d oos=%d tp_rate_oos=%.3f thr=%.3f' % (
            side, auc, trm.sum(), tem.sum(), yb[tem].mean(), thr), flush=True)

    agg = policy_backtest(labs, meta, models, t1)
    R = np.array([a['R'] for a in agg], dtype=float) if agg else np.array([])
    policy = dict(n=int(R.size), sum_R=round(float(R.sum()), 2),
                  wr=round(100.0 * float((R > 0).mean()), 1) if R.size else 0.0,
                  margin_R=round(float(R.mean()), 4) if R.size else 0.0,
                  net_full_R=round(float(sum(a['net_R'] for a in agg)), 2) if agg else 0.0)
    print('POLICY OOS: %s' % policy, flush=True)

    res = dict(measured=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               features=FEATURES, quality=quality, policy=policy)
    os.makedirs('research/results', exist_ok=True)
    json.dump(res, open('research/results/ml2.json', 'w'), indent=1, ensure_ascii=False)
    lines = ['# ML тройной барьер (раунд 8, %s UTC)' % res['measured'][:10], '']
    for side, d in quality.items():
        lines.append('- **%s**: AUC=%.4f | train=%d | OOS=%d | TP-rate OOS=%.3f | порог p=%.3f'
                     % (side, d['auc'], d['n_train'], d['n_oos'], d['tp_rate_oos'],
                        d['prob_threshold']))
    lines.append('- Политика top-20%% OOS: n=%d, %+.2fR (raw), net-full %+.2fR, wr=%.1f%%, '
                 'маржа %.4fR/сд' % (policy['n'], policy['sum_R'], policy['net_full_R'],
                                     policy['wr'], policy['margin_R']))
    open('research/results/ml2.md', 'w').write('\n'.join(lines) + '\n')
    print('saved ml2.json + md', flush=True)


if __name__ == '__main__':
    main()
