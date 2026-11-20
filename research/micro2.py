"""research/micro2.py — раунд 8: новые микропатерны (второе поколение).

Семейства (все 2-сторонние, протокол deep_sym: IS 120д / OOS 60д, monthly, net-fees):
  climax_fade   — разогретая свеча-климакс (|ret| > 3*ATR%, vol z>2.5) -> фейд
  nr_break      — 7+ узких свечей, закрытие за их диапазон -> продолжение
  vwap_z        — растяжение от VWAP >= 2.5 сигм (окно 1440) -> возврат
  vol_shock     — объёмный шок (z > 3) в направлении свечи -> продолжение
  wick_chain2   — 2 подряд лист-абсорбционных свечи -> фейд (усиленный)
  doji_exhaust  — дожи (body/range < 0.15) после серии >= 5 -> фейд

Exits: m1 (rr 1.5x2.5), m1_trail, fade20 (time 20м, sl 1.2xATR).
Виды: 1m, 15m, 60m. Симметричный набор => честный multiple-testing подсчёт.

Запуск: ./venv/bin/python research/micro2.py
"""
import datetime
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import research.lab as lab
import research.deep as deep
from research.indicators_vec import rollmax_prev, rollmin_prev
from research.scenarios import net_R, SCENARIOS

DAY = lab.DAY

MCFG = dict(atr_min_pct=0.03, vol_mult=1.0)
EXITS2 = {
    'm1': deep.DEEP_EXITS['m1'],
    'm1_trail': deep.DEEP_EXITS['m1_trail'],
    'fade20': dict(sl_atr_mult=1.2, exit='time', max_hold_min=20,
                   sl_max_pct=0.015),
}


def _zscore(x, win):
    x = np.nan_to_num(x, nan=0.0)
    cs = np.concatenate(([0.0], np.cumsum(x)))
    cs2 = np.concatenate(([0.0], np.cumsum(x * x)))
    idx = np.maximum(np.arange(x.size) - win, 0)
    n = (np.arange(1, x.size + 1) - idx).astype(float)
    mu = (cs[1:] - cs[idx]) / n
    var = np.maximum((cs2[1:] - cs2[idx]) / n - mu ** 2, 1e-18)
    return (x - mu) / np.sqrt(var)


def micro2_families(ld, cfg):
    n = ld.ts.size
    d = np.sign(ld.c - ld.o)
    rng = np.maximum(ld.h - ld.l, 1e-12)
    body = np.abs(ld.c - ld.o)
    ret1 = np.nan_to_num(ld.ret1, nan=0.0)
    atrp = np.nan_to_num(ld.atr_pct, nan=1e9) / 100.0
    amin = cfg.get('atr_min_pct', 0.03)
    base = np.isfinite(ld.atr_pct) & (ld.atr_pct >= amin)
    vr = np.where(np.isfinite(ld.vol_ratio), ld.vol_ratio, 0.0)
    vz = _zscore(np.log1p(np.where(ld.v > 0, ld.v, 1.0)), 1440)
    vwz = _zscore(ld.c / np.where(np.isfinite(ld.vwap), ld.vwap, ld.c) - 1.0, 1440)
    s, sdir = deep.streak_len(ld.c)

    out = {}
    clim = (np.abs(ret1) > 3 * atrp) & (vz > 2.5) & base
    out['climax_fade'] = (((d < 0) & clim), ((d > 0) & clim))

    narrow = (rng < 0.6 * np.abs(np.nan_to_num(ld.atr) * ld.c / ld.c) * 0 + rng)
    use_atr = np.nan_to_num(ld.atr, nan=0.0)
    narrow = (rng < 0.6 * use_atr) & (use_atr > 0)
    cnt_n = np.concatenate(([0], np.cumsum(narrow.astype(np.int64))))
    i7 = np.maximum(np.arange(n) - 7, 0)
    recent7 = (cnt_n[1:] - cnt_n[i7]) >= 7
    out['nr_break'] = ((recent7 & (ld.c > rollmax_prev(ld.h, 7)) & (vr > 1.2) & base),
                       (recent7 & (ld.c < rollmin_prev(ld.l, 7)) & (vr > 1.2) & base))

    out['vwap_z'] = ((vwz <= -2.5) & base, (vwz >= 2.5) & base)

    out['vol_shock'] = (((vz > 3.0) & (d > 0) & base),
                        ((vz > 3.0) & (d < 0) & base))

    low_wick = (np.minimum(ld.o, ld.c) - ld.l) / rng
    up_wick = (ld.h - np.maximum(ld.o, ld.c)) / rng
    w1 = (up_wick > 0.6) & (vr > 1.5)
    w2 = (low_wick > 0.6) & (vr > 1.5)
    w1p = np.roll(w1, 1); w1p[0] = False
    w2p = np.roll(w2, 1); w2p[0] = False
    out['wick_chain2'] = ((w1 & w1p) & base, (w2 & w2p) & base)

    out['doji_exhaust'] = (((s >= 5) & (sdir < 0) & (body / rng < 0.15) & base),
                           ((s >= 5) & (sdir > 0) & (body / rng < 0.15) & base))
    return out


def run_window(labs, mcache, idea, exit_cfg, lo, hi):
    trades = []
    for sym, ld in labs.items():
        key = (sym, idea)
        if key not in mcache:
            continue
        larr, sarr = mcache[key]
        mask = np.flatnonzero((larr | sarr) & (ld.ts >= lo) & (ld.ts < hi))
        next_free = -1
        for i in mask:
            if i <= next_free:
                continue
            side = 'long' if larr[i] else 'short'
            t = lab.sim(ld, exit_cfg, int(i), side, hi)
            if t is None:
                continue
            next_free = int(i) + t['bars_held']
            trades.append(t)
    return trades


def main():
    from config import SYMBOLS
    full = [s.replace('/', '').split(':')[0] for s in SYMBOLS]
    raw = {sym: lab.load_1m(sym, days=190) for sym in full}
    labs1 = {sym: lab.LabData(sym, b) for sym, b in raw.items()}
    for sym in full:
        if sym.startswith('BTC'):
            lab.BTC = labs1[sym]
    t0 = int(min(l.ts[0] for l in labs1.values()))
    t1 = int(max(l.ts[-1] for l in labs1.values()))
    is_end, oos_start = t0 + 120 * DAY, t1 - 60 * DAY
    views = [('1m', labs1), ('15m', {s: deep.build_tf_lab(s, raw[s], 15) for s in full}),
             ('60m', {s: deep.build_tf_lab(s, raw[s], 60) for s in full})]
    rows = []
    tested = 0
    for view_name, v_labs in views:
        for sym, ld in v_labs.items():
            fams = micro2_families(ld, MCFG)
            mc = {(sym, k): v for k, v in fams.items()}
            for idea, (la, sa) in fams.items():
                n_sig = int((la | sa).sum())
                if n_sig < 30:
                    continue
                for ex_name, ex in EXITS2.items():
                    tested += 1
                    tr_is = run_window({sym: ld}, mc, idea, ex, t0, is_end)
                    st_i = lab.stats(tr_is)
                    if st_i['n'] < 20 or st_i['sum_R'] <= 0.5:
                        continue
                    acc_is, _ = lab.portfolio_accept(tr_is, ex, is_end)
                    st_i = lab.stats(acc_is)
                    tr_o = run_window({sym: ld}, mc, idea, ex, oos_start, t1)
                    acc_o, eq = lab.portfolio_accept(tr_o, ex, t1)
                    st_o = lab.stats(acc_o)
                    row = dict(view=view_name, symbol=sym, idea=idea, exit=ex_name,
                               is_n=st_i['n'], is_sum_R=st_i['sum_R'],
                               oos_n=st_o['n'], oos_sum_R=st_o['sum_R'],
                               oos_wr=st_o['win_rate'], oos_margin=round(
                                   st_o['sum_R'] / st_o['n'], 3) if st_o['n'] else None)
                    if st_o['n'] >= 12:
                        row['monthly'] = lab.monthly_table(acc_o)
                        row['oos_net_full'] = round(sum(
                            net_R(t, mk, tk) for t in acc_o
                            for mk, tk in [list(SCENARIOS.values())[0]]), 2)
                    rows.append(row)
        print('[view %s done] passes %d rows %d' % (view_name, tested, len(rows)),
              flush=True)

    survivors = [r for r in rows
                 if r['oos_sum_R'] > 0 and (r.get('oos_margin') or 0) >= 0.05
                 and r['oos_n'] >= 12]
    survivors.sort(key=lambda r: r['oos_sum_R'], reverse=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    os.makedirs('research/results', exist_ok=True)
    json.dump(dict(measured=stamp, tested_combos=tested, rows=len(rows),
                   survivors=survivors, all=rows),
              open('research/results/micro2.json', 'w'), indent=1, ensure_ascii=False)
    lines = ['# Микропатерны-2 (раунд 8, %s UTC)' % stamp[:10], '',
             'Всего прожато комбинаций: %d, IS-пасс: %d, OOS-выживших (маржа>=0.05R/сд, n>=12): %d'
             % (tested, len(rows), len(survivors)), '']
    if survivors:
        lines += ['| Вид | Монета | Семейство | Выход | IS | OOS | Маржа R/сд | мес+ |',
                  '|-----|--------|-----------|-------|----|----|------------|------|']
        for r in survivors[:20]:
            mon = r.get('monthly') or {}
            mp = sum(1 for v in mon.values() if v['sum_R'] > 0)
            lines.append('| %s | %s | %s | %s | %+.1fR/%d | %+.1fR/%d | %.3f | %d/%d |' % (
                r['view'], r['symbol'].replace('USDT', ''), r['idea'], r['exit'],
                r['is_sum_R'], r['is_n'], r['oos_sum_R'], r['oos_n'],
                r['oos_margin'], mp, len(mon)))
    else:
        lines.append('**Выживших нет.**')
    open('research/results/micro2.md', 'w').write('\n'.join(lines) + '\n')
    print('saved micro2.json + md; выживших: %d / %d' % (len(survivors), len(rows)),
          flush=True)


if __name__ == '__main__':
    main()
