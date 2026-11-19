"""research/deep.py — глубокий поиск: микроструктура 1m + крупные ТФ (15m/1h).

Семейства:
- микроструктура 1m: streak3 (серия закрытий), heat (разогрев тел+объём),
  wick_absorb (адсорбция хвостом), micro_don5 (микропробой), streak_rev (фейд серии)
- классики из lab.py на TF 15m/1h с широкими стопами (издержки/стоп ~0.05R)

Протокол: IS 120д -> OOS 60д; выжившие проверяются при комиссиях maker-maker 4бп.
Запуск: ./venv/bin/python research/deep.py [--quick]
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import research.lab as lab
from research.data import load_1m
from research.indicators_vec import rollmax_prev, rollmin_prev
from scenarios import net_R, SCENARIOS

DAY = lab.DAY


def streak_len(c):
    """Длина серии одинаковых направлений закрытия (>=0)."""
    d = np.sign(np.diff(c, prepend=c[0]))
    s = np.zeros(c.size, dtype=np.int64)
    cur = 0
    for i in range(1, c.size):
        cur = cur + 1 if d[i] == d[i - 1] and d[i] != 0 else (1 if d[i] != 0 else 0)
        s[i] = cur
    return s, d


def micro_families(ld, cfg):
    """Микроструктурные семейства на 1m: idea -> (long_arr, short_arr)."""
    n = ld.ts.size
    base = np.isfinite(ld.atr_pct) & (ld.atr_pct >= cfg.get('atr_min_pct', 0.03))
    vr = np.where(np.isfinite(ld.vol_ratio), ld.vol_ratio, 0.0)
    vol_ok = vr >= cfg.get('vol_mult', 1.5)
    trend1h = ld.trend['1h']
    valid = trend1h['map'] >= 0
    j = np.where(valid, trend1h['map'], 0)
    up1h = valid & (trend1h['c'][j] > trend1h['e20'][j]) & (trend1h['e20'][j] > trend1h['e50'][j])
    dn1h = valid & (trend1h['c'][j] < trend1h['e20'][j]) & (trend1h['e20'][j] < trend1h['e50'][j])
    s, d = streak_len(ld.c)
    body = np.abs(ld.c - ld.o)
    rng = np.maximum(ld.h - ld.l, 1e-12)
    out = {}
    # 1. streak3: серия из 3+ закрытий в одну сторону (продолжение)
    s3 = (s >= 3) & (d > 0)
    s3d = (s >= 3) & (d < 0)
    out['streak3'] = ((s3 & vol_ok & base), (s3d & vol_ok & base))
    # 2. heat: 3 свечи растущих тел + растущий объём (разогрев)
    body_up = (body > np.roll(body, 1)) & (np.roll(body, 1) > np.roll(body, 2))
    v_up = (ld.v > np.roll(ld.v, 1)) & (np.roll(ld.v, 1) > np.roll(ld.v, 2))
    heat_l = (d > 0) & body_up & v_up & (vr > 1.2)
    heat_s = (d < 0) & body_up & v_up & (vr > 1.2)
    out['heat'] = ((heat_l & base), (heat_s & base))
    # 3. wick_absorb: хвост >=60% диапазона против движения + объём
    low_wick = (np.minimum(ld.o, ld.c) - ld.l) / rng
    up_wick = (ld.h - np.maximum(ld.o, ld.c)) / rng
    out['wick_absorb'] = ((up_wick > 0.6) & (vr > 1.5) & base,
                          (low_wick > 0.6) & (vr > 1.5) & base)
    # 4. micro_don5: микро-пробой 5 баров
    out['micro_don5'] = ((ld.c > rollmax_prev(ld.h, 5)) & vol_ok & base,
                         (ld.c < rollmin_prev(ld.l, 5)) & vol_ok & base)
    # 5. streak_rev: фейд серии >=4 против тренда 1h
    out['streak_rev'] = (((s >= 4) & (d < 0) & up1h & base),
                         ((s >= 4) & (d > 0) & dn1h & base))
    # 6. streak_trend: серия в тренд 1h
    out['streak_trend'] = (((s >= 3) & (d > 0) & up1h & base),
                           ((s >= 3) & (d < 0) & dn1h & base))
    # 7. heat_trend: разогрев в тренд 1h
    out['heat_trend'] = ((heat_l & up1h & base), (heat_s & dn1h & base))
    return out


DEEP_EXITS = {
    'm1': dict(sl_atr_mult=1.5, rr=2.5, exit='rr'),
    'm1_trail': dict(sl_atr_mult=2.0, trail_atr=2.0, exit='trail'),
    'wide:3.5x3.0': dict(sl_atr_mult=3.5, rr=3.0, exit='rr', sl_max_pct=0.02),
    'tf_trail': dict(sl_atr_mult=3.0, trail_atr=2.5, exit='trail', sl_max_pct=0.03,
                     time_stop_min=480),
    'tf_wide': dict(sl_atr_mult=4.0, rr=4.0, exit='rr', sl_max_pct=0.04,
                    time_stop_min=960, be_trigger=1.5),
}


def build_tf_lab(sym, bars1m, minutes):
    rows = lab.to_tf([(int(b[0]), b[1], b[2], b[3], b[4], b[5]) for b in bars1m], minutes)
    return lab.LabData(sym, rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()

    from config import SYMBOLS
    full = [s.replace('/', '').split(':')[0] for s in SYMBOLS]
    syms = [s for s in full if s.startswith(('BTC', 'SOL'))] if args.quick else full

    raw = {sym: load_1m(sym, days=190) for sym in syms}
    labs = {sym: lab.LabData(sym, bars) for sym, bars in raw.items()}
    for sym in syms:
        if sym.startswith('BTC'):
            lab.BTC = labs[sym]
    t0 = int(min(l.ts[0] for l in labs.values()))
    t1 = int(max(l.ts[-1] for l in labs.values()))
    is_end, oos_start = t0 + 120 * DAY, t1 - 60 * DAY
    print('IS %s..%s OOS %s..%s' % (
        datetime.datetime.fromtimestamp(t0 / 1000, datetime.timezone.utc).strftime('%Y-%m-%d'),
        datetime.datetime.fromtimestamp(is_end / 1000, datetime.timezone.utc).strftime('%Y-%m-%d'),
        datetime.datetime.fromtimestamp(oos_start / 1000, datetime.timezone.utc).strftime('%Y-%m-%d'),
        datetime.datetime.fromtimestamp(t1 / 1000, datetime.timezone.utc).strftime('%Y-%m-%d')),
          flush=True)

    views = [('1m', labs, lab.MASK_CFGS, lab.masks)]
    for tf, minutes in (('15m', 15), ('1h', 60)):
        tf_labs = {sym: build_tf_lab(sym, raw[sym], minutes) for sym in syms}
        mc = dict(atr_min_pct=0.03, vol_mult=1.3)
        views.append((tf, tf_labs, {'tf': mc}, lab.masks))

    results = []
    for view_name, v_labs, vcfgs, fam in views:
        exits = DEEP_EXITS if view_name == '1m' else {
            'wide:3.5x3.0': DEEP_EXITS['wide:3.5x3.0'],
            'tf_trail': DEEP_EXITS['tf_trail'],
            'tf_wide': DEEP_EXITS['tf_wide'],
        }
        for mc_name, mcfg in vcfgs.items():
            mc_all = {}
            for sym, ld in v_labs.items():
                if view_name == '1m' and mc_name in ('vol1.5', 'vol1.8', 'loose'):
                    fams = dict(lab.masks(ld, mcfg))
                    fams.update(micro_families(ld, mcfg))
                else:
                    fams = fam(ld, mcfg)
                for idea, (la, sa) in fams.items():
                    mc_all[(sym, idea)] = (la, sa)
            ideas = sorted({k[1] for k in mc_all})
            for idea in ideas:
                for ex_name, ex in exits.items():
                    tr_is = lab.run_window(v_labs, mc_all, idea, ex, t0, is_end)
                    acc_is, _ = lab.portfolio_accept(tr_is, ex, is_end)
                    st = lab.stats(acc_is)
                    row = dict(view=view_name, mask=mc_name, idea=idea, exit=ex_name, IS=st)
                    if st['n'] >= 30 and st['sum_R'] > 0:
                        acc_oos, _ = lab.portfolio_accept(
                            lab.run_window(v_labs, mc_all, idea, ex, oos_start, t1), ex, t1)
                        row['OOS'] = lab.stats(acc_oos)
                        if row['OOS']['n'] >= 20:
                            oos_scen = {name: round(sum(
                                net_R(t, mk, tk) for t in acc_oos), 2)
                                for name, (mk, tk) in SCENARIOS.items()}
                            row['OOS_net'] = oos_scen
                            row['monthly'] = lab.monthly_table(acc_oos)
                    results.append(row)
                    o = row.get('OOS')
                    print('[%s|%s|%s|%s] IS n=%d %.1fR | OOS %s' % (
                        view_name, mc_name, idea, ex_name, st['n'], st['sum_R'],
                        ('n=%d %.1fR mm=%s' % (o['n'], o['sum_R'],
                                               row.get('OOS_net', {}).get('maker-maker 4bp')))
                        if o else '-'), flush=True)

    oos_ok = [r for r in results if r.get('OOS') and r['OOS']['sum_R'] > 0]
    oos_ok.sort(key=lambda r: r['OOS']['sum_R'], reverse=True)
    surv = [r for r in oos_ok if r.get('OOS_net', {}).get('maker-maker 4bp', -1) > 0]
    os.makedirs('research/results', exist_ok=True)
    json.dump(dict(results=results, oos_ok=oos_ok[:30]),
              open('research/results/deep.json', 'w'), indent=1, ensure_ascii=False)

    rep = ['# Глубокий коп (deep.py): микроструктура 1m + ТФ 15m/1h\n']
    rep.append('Прогонов: %d. OOS+ : %d. Выживают при maker-maker 4бп: %d.\n' % (
        len(results), len(oos_ok), len(surv)))
    rep.append('| view | mask | idea | exit | IS n/sumR | OOS n/sumR | OOS mm4 | OOS full | месяцы OOS |')
    rep.append('|---|---|---|---|---|---|---|---|---|')
    for r in (surv + oos_ok)[:20]:
        o, m = r['OOS'], r.get('monthly', {})
        pos = sum(1 for v in m.values() if v['sum_R'] > 0)
        rep.append('| %s | %s | %s | %s | %d/%.1f | %d/%.1f | %s | %s | %d/%d |' % (
            r['view'], r['mask'], r['idea'], r['exit'], r['IS']['n'], r['IS']['sum_R'],
            o['n'], o['sum_R'], r.get('OOS_net', {}).get('maker-maker 4bp', '-'),
            r.get('OOS_net', {}).get('full(m+t 7.5bp)', '-'), pos, len(m)))
    with open('research/results/deep.md', 'w') as f:
        f.write('\n'.join(rep) + '\n')
    print('done. oos_ok=%d surv=%d' % (len(oos_ok), len(surv)), flush=True)


if __name__ == '__main__':
    main()
