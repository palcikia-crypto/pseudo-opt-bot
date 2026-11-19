"""research/combo.py — рандомные комбинации семейств сигналов + IS/OOS-валидация.

Сэмплируем пары/тройки семейств (union масок) на ТФ 15m/1h и микроструктуре 1m,
гоняем портфель по всем монетам: IS 120д -> отбор, OOS 60д -> проверка,
maker-maker и full-комиссии на OOS, месяцы.

Запуск: ./venv/bin/python research/combo.py [--n 40 --seed 7]
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import research.lab as lab
import research.deep as deep
from research.scenarios import net_R, SCENARIOS

DAY = lab.DAY
FAM_TF = ('trend_bb15', 'trend_bb1h', 'squeeze', 'mr_rsi5', 'mr_vwap',
          'donchian', 'impulse', 'pullback', 'btc_led')
FAM_M1 = ('streak3', 'heat', 'wick_absorb', 'micro_don5', 'streak_rev',
          'streak_trend', 'heat_trend')
EXITS_TF = {k: deep.DEEP_EXITS[k] for k in ('tf_trail', 'tf_wide')}
EXITS_M1 = {k: deep.DEEP_EXITS[k] for k in ('m1', 'm1_trail', 'wide:3.5x3.0')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    from config import SYMBOLS
    full = [s.replace('/', '').split(':')[0] for s in SYMBOLS]
    raw = {s: lab.load_1m(s, days=190) for s in full}
    t0 = int(min(min(b[0] for b in v) for v in raw.values()))
    t1 = int(max(max(b[0] for b in v) for v in raw.values()))
    is_end, oos_start = t0 + 120 * DAY, t1 - 60 * DAY

    views = []
    for tf, minutes in (('15m', 15), ('1h', 60)):
        vl = {s: deep.build_tf_lab(s, raw[s], minutes) for s in full}
        fams = {}
        for s, ld in vl.items():
            if s.startswith('BTC'):
                lab.BTC = ld
            fams.update({(s, idea): m for idea, m in lab.masks(ld, dict(
                atr_min_pct=0.03, vol_mult=1.3)).items()})
        views.append((tf, vl, fams, FAM_TF, EXITS_TF))
    v1 = {s: lab.LabData(s, raw[s]) for s in full}
    for s in full:
        if s.startswith('BTC'):
            lab.BTC = v1[s]
    fams1 = {}
    for s, ld in v1.items():
        M = lab.masks(ld, dict(atr_min_pct=0.03, vol_mult=1.3))
        M.update(deep.micro_families(ld, dict(atr_min_pct=0.03, vol_mult=1.3)))
        fams1.update({(s, i): m for i, m in M.items()})
    views.append(('1m', v1, fams1, FAM_M1, EXITS_M1))

    results = []
    for view, vl, fams, fam_list, exits in views:
        for _ in range(args.n // 3):
            k = rng.choice((2, 3))
            ideas = tuple(sorted(rng.sample(fam_list, k)))
            ex_name = rng.choice(list(exits))
            ex = exits[ex_name]
            mc = {key: m for key, m in fams.items() if key[1] in ideas}
            tr_is = []
            for idea in ideas:
                tr_is += lab.run_window(vl, mc, idea, ex, t0, is_end)
            acc_is, _ = lab.portfolio_accept(tr_is, ex, is_end)
            si = lab.stats(acc_is)
            row = dict(view=view, combo='+'.join(ideas), exit=ex_name,
                       is_n=si['n'], is_sum_R=si['sum_R'])
            if si['n'] >= 30 and si['sum_R'] > 0:
                tr_o = []
                for idea in ideas:
                    tr_o += lab.run_window(vl, mc, idea, ex, oos_start, t1)
                acc_o, _ = lab.portfolio_accept(tr_o, ex, t1)
                so = lab.stats(acc_o)
                row.update(oos_n=so['n'], oos_sum_R=so['sum_R'], oos_wr=so['win_rate'])
                row['oos_net'] = {name: round(sum(net_R(t, mk, tk) for t in acc_o), 2)
                                  for name, (mk, tk) in SCENARIOS.items()}
                row['monthly'] = lab.monthly_table(acc_o)
            results.append(row)
            print('[%s] %s | %s IS n=%d %.1fR%s' % (
                view, row['combo'], ex_name, si['n'], si['sum_R'],
                (' | OOS n=%d %.1fR mm=%.1f' % (
                    row['oos_n'], row['oos_sum_R'],
                    row.get('oos_net', {}).get('maker-maker 4bp', 0)))
                if 'oos_n' in row else ''), flush=True)

    ok = [r for r in results if r.get('oos_sum_R', 0) > 0
          and r.get('oos_net', {}).get('maker-maker 4bp', -1) > 0]
    ok.sort(key=lambda r: r['oos_sum_R'], reverse=True)
    write_report(results, ok)


def write_report(results, ok):
    os.makedirs('research/results', exist_ok=True)
    json.dump(dict(results=results, survivors=ok),
              open('research/results/combo.json', 'w'), indent=1, ensure_ascii=False)
    rep = ['# Рандомные комбинации семейств (combo.py)\n',
           'Прогонов: %d. Прошли IS+ и OOS+ при maker-maker 4бп: %d.\n' % (
               len(results), len(ok)),
           '| view | combo | exit | IS n/sumR | OOS n/sumR | OOS mm4 | OOS full | месяцы |',
           '|---|---|---|---|---|---|---|---|']
    shown = 0
    for r in (ok + [x for x in results if 'oos_n' not in x]):
        if shown >= 25:
            break
        m = r.get('monthly', {})
        pos = sum(1 for v in m.values() if v['sum_R'] > 0)
        rep.append('| %s | %s | %s | %d/%.1f | %s/%s | %s | %s | %d/%d |' % (
            r['view'], r['combo'], r['exit'], r['is_n'], r['is_sum_R'],
            r.get('oos_n', '-'), r.get('oos_sum_R', '-'),
            r.get('oos_net', {}).get('maker-maker 4bp', '-'),
            r.get('oos_net', {}).get('full(m+t 7.5bp)', '-'), pos, len(m)))
        shown += 1
    with open('research/results/combo.md', 'w') as f:
        f.write('\n'.join(rep) + '\n')
    print('done. survivors=%d' % len(ok), flush=True)


if __name__ == '__main__':
    main()
