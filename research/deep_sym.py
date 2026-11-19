"""research/deep_sym.py — per-symbol скан на ТФ 15m/1h + микроструктура 1m.

Ищет монеты с локальным сырым эджем; OOS-победителей проверяет на
maker-maker 4бп и full-комиссиях; пишет monthly-стабильность OOS.

Запуск: ./venv/bin/python research/deep_sym.py
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


def main():
    from config import SYMBOLS
    full = [s.replace('/', '').split(':')[0] for s in SYMBOLS]
    raw = {sym: lab.load_1m(sym, days=190) for sym in full}
    labs = {sym: lab.LabData(sym, b) for sym, b in raw.items()}
    for sym in full:
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

    views = []
    for tf, minutes in (('15m', 15), ('1h', 60)):
        views.append((tf, {sym: deep.build_tf_lab(sym, raw[sym], minutes) for sym in full},
                      deep.DEEP_EXITS))
    views.append(('1m', labs,
                  {k: v for k, v in deep.DEEP_EXITS.items() if not k.startswith('tf_')}))

    rows = []
    for view_name, v_labs, exits in views:
        for sym, ld in v_labs.items():
            v1 = {sym: ld}
            mcfg = dict(atr_min_pct=0.03, vol_mult=1.3)
            fams = dict(lab.masks(ld, mcfg)) if view_name != '1m' \
                else dict(lab.masks(ld, mcfg), **deep.micro_families(ld, mcfg))
            mc_all = {(sym, idea): (la, sa) for idea, (la, sa) in fams.items()}
            for idea, (la, sa) in fams.items():
                for ex_name, ex in exits.items():
                    acc_is, _ = lab.portfolio_accept(
                        lab.run_window(v1, mc_all, idea, ex, t0, is_end), ex, is_end)
                    st = lab.stats(acc_is)
                    if st['n'] < 20 or st['sum_R'] <= 0:
                        continue
                    acc_oos, _ = lab.portfolio_accept(
                        lab.run_window(v1, mc_all, idea, ex, oos_start, t1), ex, t1)
                    st_o = lab.stats(acc_oos)
                    row = dict(view=view_name, symbol=sym, idea=idea, exit=ex_name,
                               is_n=st['n'], is_sum_R=st['sum_R'],
                               oos_n=st_o['n'], oos_sum_R=st_o['sum_R'],
                               oos_wr=st_o['win_rate'])
                    if st_o['n'] >= 15:
                        row['oos_net'] = {name: round(sum(
                            net_R(t, mk, tk) for t in acc_oos), 2)
                            for name, (mk, tk) in SCENARIOS.items()}
                        row['monthly'] = lab.monthly_table(acc_oos)
                    rows.append(row)
        print('[view done] %s (rows %d)' % (view_name, len(rows)), flush=True)

    oos_raw = sorted([r for r in rows if r['oos_sum_R'] > 0],
                     key=lambda r: r['oos_sum_R'], reverse=True)
    surv = [r for r in oos_raw if r.get('oos_net', {}).get('maker-maker 4bp', -1) > 0]
    os.makedirs('research/results', exist_ok=True)
    json.dump(dict(rows=rows, oos_raw_positive=oos_raw[:40]),
              open('research/results/deep_sym.json', 'w'), indent=1, ensure_ascii=False)

    rep = ['# Per-symbol глубокий коп (deep_sym.py)\n']
    rep.append('Всего IS+ вариантов: %d. OOS+: %d. Выживают при maker-maker 4бп: %d.\n' % (
        len(rows), len(oos_raw), len(surv)))
    rep.append('| view | symbol | idea | exit | IS n/sumR | OOS n/sumR | WR% | OOS mm4 | OOS full | месяцы |')
    rep.append('|---|---|---|---|---|---|---|---|---|---|')
    for r in oos_raw[:25]:
        m = r.get('monthly', {})
        pos = sum(1 for v in m.values() if v['sum_R'] > 0)
        rep.append('| %s | %s | %s | %s | %d/%.1f | %d/%.1f | %.0f | %s | %s | %d/%d |' % (
            r['view'], r['symbol'], r['idea'], r['exit'], r['is_n'], r['is_sum_R'],
            r['oos_n'], r['oos_sum_R'], r['oos_wr'],
            r.get('oos_net', {}).get('maker-maker 4bp', '-'),
            r.get('oos_net', {}).get('full(m+t 7.5bp)', '-'), pos, len(m)))
    if surv:
        rep.append('\n## Кандидаты maker-maker (после комиссий в плюсе)\n')
        for r in surv[:10]:
            m = r.get('monthly', {})
            rep.append('- **%s/%s %s**: OOS %d сделок, raw %.1fR, mm4 %+.1fR, full %+.1fR, '
                       'месяцев в плюсе %d/%d' % (
                           r['symbol'], r['view'], r['idea'], r['oos_n'], r['oos_sum_R'],
                           r['oos_net'].get('maker-maker 4bp'),
                           r['oos_net'].get('full(m+t 7.5bp)'),
                           sum(1 for v in m.values() if v['sum_R'] > 0), len(m)))
    with open('research/results/deep_sym.md', 'w') as f:
        f.write('\n'.join(rep) + '\n')
    print('done. is+=%d oos+=%d surv=%d' % (len(rows), len(oos_raw), len(surv)), flush=True)


if __name__ == '__main__':
    main()
