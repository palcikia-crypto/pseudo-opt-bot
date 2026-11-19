"""research/regime.py — режимная атрибуция и BTC-фильтры для живых кандидатов.

Кандидаты: APTUSDT 15m mr_rsi5 + tf_trail; XRPUSDT 1h mr_vwap + m1_trail
(конфиг deep_sym: atr_min 0.03, vol_mult 1.3). Вопросы:
  1) В каких BTC-режимах кандидаты реально зарабатывают (атрибуция: тренд 1h,
     вола BTC выше/ниже медианы, импульс BTC того же знака)?
  2) Улучшают ли явные BTC-гейты результат (3 варианта на кандидата, IS/OOS)?

Запуск: ./venv/bin/python research/regime.py
"""
import datetime
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import research.lab as lab
import research.deep as deep
from research.scenarios import net_R, SCENARIOS

DAY = lab.DAY

CANDIDATES = [
    dict(name='APT_mr_rsi5', sym='APTUSDT', tf=15, idea='mr_rsi5', exit='tf_trail'),
    dict(name='XRP_mr_vwap', sym='XRPUSDT', tf=60, idea='mr_vwap', exit='m1_trail'),
]
MCFG = dict(atr_min_pct=0.03, vol_mult=1.3)

GATES = ('trend_side', 'btc_vol_hi', 'btc_impulse')


def btc_features(btcl, ts_ref):
    pos = np.searchsorted(btcl.ts, ts_ref)
    pv = np.minimum(pos, btcl.ts.size - 1)
    valid = (pos < btcl.ts.size) & (btcl.ts[pv] == ts_ref)
    j = pv
    ret60 = np.array([float(np.nansum(btcl.ret1[max(int(x) - 60, 0):int(x)]))
                      for x in j])
    atrp = np.array([btcl.atr_pct[int(x)] if valid[k] else np.nan
                     for k, x in enumerate(j)])
    t1h = btcl.trend['1h']
    m = t1h['map']
    tv = m >= 0
    jj = np.where(tv, m, 0)
    up = t1h['c'][jj] > t1h['e20'][jj]
    dn = t1h['c'][jj] < t1h['e20'][jj]
    tr_val = np.where(up, 1, np.where(dn, -1, 0))
    tr = np.where(valid & tv[j], tr_val[j], 0)
    atr_med = float(np.nanmedian(atrp[np.isfinite(atrp)]))
    return dict(tr=tr, atrp=atrp, atr_med=atr_med, ret60=ret60, valid=valid)


def _sim_all(ld, la, sa, ex):
    trades = []
    mask = np.flatnonzero(la | sa)
    for i in mask:
        side = 'long' if la[i] else 'short'
        t = lab.sim(ld, ex, int(i), side, None)
        if t is not None:
            trades.append(t)
    return trades


def gate_mask(gate, la, sa, f):
    if gate is None:
        return la, sa
    la2 = la.copy()
    sa2 = sa.copy()
    if gate == 'trend_side':
        la2 &= (f['tr'] == 1)
        sa2 &= (f['tr'] == -1)
    elif gate == 'btc_vol_hi':
        hi = f['atrp'] > f['atr_med']
        la2 &= hi
        sa2 &= hi
    elif gate == 'btc_impulse':
        direction = np.where(f['ret60'] > 0, 1, np.where(f['ret60'] < 0, -1, 0))
        la2 &= (direction == 1)
        sa2 &= (direction == -1)
    return la2, sa2


def window_stats(trades, lo, hi):
    sel = [t for t in trades if t['ts_entry'] >= lo and t['ts_entry'] < hi]
    st = lab.stats(sel)
    mon = lab.monthly_table(sel) if sel else {}
    st['months_pos'] = sum(1 for v in mon.values() if v['sum_R'] > 0) if mon else None
    st['months'] = len(mon)
    return st, sel


def main():
    from config import SYMBOLS
    full = [s.replace('/', '').split(':')[0] for s in SYMBOLS]
    needed = {'BTCUSDT'} | {c['sym'] for c in CANDIDATES}
    raw_all = {sym: lab.load_1m(sym, days=190) for sym in full if sym in needed}
    btcl = lab.LabData('BTCUSDT', raw_all['BTCUSDT'])
    t0 = int(min(raw_all[c['sym']][0][0] for c in CANDIDATES))
    t1 = int(max(raw_all[c['sym']][-1][0] for c in CANDIDATES))
    is_end, oos_start = t0 + 120 * DAY, t1 - 60 * DAY

    report = {}
    for cand in CANDIDATES:
        ld = deep.build_tf_lab(cand['sym'], raw_all[cand['sym']], cand['tf'])
        M = lab.masks(ld, MCFG)
        la, sa = M[cand['idea']]
        ex = deep.DEEP_EXITS[cand['exit']]
        f = btc_features(btcl, ld.ts)
        trades = _sim_all(ld, la, sa, ex)

        st_is, sel_is = window_stats(trades, t0, is_end)
        st_oos, sel_oos = window_stats(trades, oos_start, t1)

        segs = {}
        for t in trades:
            k = int(np.searchsorted(ld.ts, t['ts_entry']) - 1)
            k = max(min(k, ld.ts.size - 1), 0)
            seg_t = {1: 'btc_up', -1: 'btc_dn', 0: 'btc_chop'}[int(f['tr'][k])]
            seg_v = 'vol_hi' if (np.isfinite(f['atrp'][k])
                                 and f['atrp'][k] > f['atr_med']) else 'vol_lo'
            imp = 1 if f['ret60'][k] > 0 else (-1 if f['ret60'][k] < 0 else 0)
            side = 1 if t['side'] == 'long' else -1
            seg_i = 'same_dir' if imp == side else 'other_dir'
            for key in (seg_t, seg_v, seg_i, seg_t + '+' + seg_v,
                        seg_v + '+' + seg_i, seg_t + '+' + seg_i):
                segs.setdefault(key, []).append(t)
        seg_table = {k: dict(n=len(v), sum_R=round(sum(x['R'] for x in v), 2),
                             wr=round(100.0 * sum(1 for x in v if x['R'] > 0) / len(v), 1))
                     for k, v in sorted(segs.items()) if len(v) >= 10}

        gates = {}
        for g in GATES:
            la2, sa2 = gate_mask(g, la, sa, f)
            g_trades = _sim_all(ld, la2, sa2, ex)
            g_is, _ = window_stats(g_trades, t0, is_end)
            g_oos, sel_g = window_stats(g_trades, oos_start, t1)
            mk, tk = list(SCENARIOS.values())[-1]
            g_oos['net_full_fee'] = round(sum(net_R(t, mk, tk) for t in sel_g), 2) \
                if sel_g else None
            gates[g] = dict(IS=g_is, OOS=g_oos)

        report[cand['name']] = dict(
            symbol=cand['sym'], tf=cand['tf'], idea=cand['idea'], exit=cand['exit'],
            baseline=dict(IS=st_is, OOS=st_oos), segments=seg_table, gates=gates)
        print('=== %s — базис IS n=%s %+.1fR | OOS n=%s %+.1fR' % (
            cand['name'], st_is['n'], st_is['sum_R'], st_oos['n'], st_oos['sum_R']),
            flush=True)
        for k, v in seg_table.items():
            print('    seg %-24s n=%4d %+.2fR wr=%.1f' % (k, v['n'], v['sum_R'], v['wr']),
                  flush=True)
        for g, gv in gates.items():
            print('    gate %-12s IS n=%s %+.1fR | OOS n=%s %+.1fR' % (
                g, gv['IS']['n'], gv['IS']['sum_R'], gv['OOS']['n'], gv['OOS']['sum_R']),
                flush=True)

    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    os.makedirs('research/results', exist_ok=True)
    json.dump(dict(measured=stamp, report=report),
              open('research/results/regime.json', 'w'), indent=1, ensure_ascii=False)

    lines = ['# BTC-режимы кандидатов (атрибуция + гейты, %s UTC)' % stamp[:10], '']
    for name, r in report.items():
        b = r['baseline']
        lines.append('## %s (%s %dm %s+%s)' % (name, r['symbol'], r['tf'],
                                               r['idea'], r['exit']))
        lines.append('- Базис: IS n=%s %+.1fR wr=%.0f%% | OOS n=%s %+.1fR wr=%.0f%%' % (
            b['IS']['n'], b['IS']['sum_R'], b['IS']['win_rate'],
            b['OOS']['n'], b['OOS']['sum_R'], b['OOS']['win_rate']))
        segs_s = ', '.join(
            '%s n=%d %+.1fR' % (k, v['n'], v['sum_R']) for k, v in
            sorted(r['segments'].items(), key=lambda kv: kv[1]['sum_R'], reverse=True)
            if v['n'] >= 30)
        lines.append('- Сегменты: %s' % (segs_s or '—'))
        lines.append('- Гейты: ' + '; '.join(
            '%s OOS %+.1fR/n=%d' % (g, gv['OOS'].get('sum_R', 0.0), gv['OOS'].get('n', 0))
            for g, gv in r['gates'].items()))
        lines.append('')
    open('research/results/regime.md', 'w').write('\n'.join(lines) + '\n')
    print('saved research/results/regime.json + .md', flush=True)


if __name__ == '__main__':
    main()
