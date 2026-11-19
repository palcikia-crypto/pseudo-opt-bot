"""research/scenarios.py — сценарный анализ: сырой (без комиссий) эдж топ-вариантов,
его устойчивость IS/OOS и выживание при трёх уровнях комиссий.

Запуск: ./venv/bin/python research/scenarios.py
"""
import datetime
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import research.lab as lab
from config import SYMBOLS

DAY = lab.DAY


def net_R(t, maker, taker):
    dirn = 1.0 if t['side'] == 'long' else -1.0
    fee_out = maker if t['exit_reason'] == 'tp' else taker
    fees = lab.MAKER + fee_out * t['exit'] / t['entry']
    return (dirn * (t['exit'] - t['entry']) - t['entry'] * fees) / (t['entry'] * t['sl_pct'])


SCENARIOS = {
    'full(m+t 7.5bp)': (0.0002, 0.00055),
    'maker-maker 4bp': (0.0002, 0.0002),
    'vip maker-t 5bp': (0.0001, 0.0004),
}


def main():
    lab.MAKER, lab.TAKER = 0.0, 0.0
    modules = [s.replace('/', '').split(':')[0] for s in SYMBOLS]
    labs = {}
    for sym in modules:
        labs[sym] = lab.LabData(sym, lab.load_1m(sym))
        if sym.startswith('BTC'):
            lab.BTC = labs[sym]
    t0 = int(min(ld.ts[0] for ld in labs.values()))
    t1 = int(max(ld.ts[-1] for ld in labs.values()))
    is_end = t0 + 120 * DAY
    oos_start = t1 - 60 * DAY
    print('IS %s .. %s | OOS %s .. %s' % (
        datetime.datetime.fromtimestamp(t0 / 1000, datetime.timezone.utc).strftime('%Y-%m-%d'),
        datetime.datetime.fromtimestamp(is_end / 1000, datetime.timezone.utc).strftime('%Y-%m-%d'),
        datetime.datetime.fromtimestamp(oos_start / 1000, datetime.timezone.utc).strftime('%Y-%m-%d'),
        datetime.datetime.fromtimestamp(t1 / 1000, datetime.timezone.utc).strftime('%Y-%m-%d')),
          flush=True)

    rows = []
    for sym, ld in labs.items():
        labs1 = {sym: ld}
        for mc_name, mcfg in lab.MASK_CFGS.items():
            M = lab.masks(ld, mcfg)
            for idea, (la, sa) in M.items():
                mc = {(sym, idea): (la, sa)}
                for ex_name, ex in lab.EXIT_CFGS.items():
                    acc_is, _ = lab.portfolio_accept(
                        lab.run_window(labs1, mc, idea, ex, t0, is_end), ex, is_end)
                    is_sum = sum(t['R'] for t in acc_is)
                    is_n = len(acc_is)
                    if is_n < 25 or is_sum <= 0:
                        continue
                    tr_oos = lab.run_window(labs1, mc, idea, ex, oos_start, t1)
                    acc_oos, _ = lab.portfolio_accept(tr_oos, ex, t1)
                    oos_n = len(acc_oos)
                    oos_raw = sum(t['R'] for t in acc_oos)
                    scen = {}
                    for sname, (mk, tk) in SCENARIOS.items():
                        scen[sname] = round(sum(net_R(t, mk, tk) for t in acc_oos), 2)
                    rows.append(dict(symbol=sym, mask=mc_name, idea=idea, exit=ex_name,
                                     is_n=is_n, is_sum_R=round(is_sum, 2),
                                     oos_n=oos_n, oos_raw_sumR=round(oos_raw, 2),
                                     oos_net=scen))
        print('[done] %s (raw IS+ kept: %d)' % (sym, len(rows)), flush=True)

    rows.sort(key=lambda r: r['oos_raw_sumR'], reverse=True)
    oos_raw_pos = [r for r in rows if r['oos_raw_sumR'] > 0]
    surv = [r for r in oos_raw_pos
            if r['oos_net']['maker-maker 4bp'] > 0 or r['oos_net']['vip maker-t 5bp'] > 0]
    os.makedirs('research/results', exist_ok=True)
    json.dump(dict(raw_is_oos_positive=oos_raw_pos, all_raw_is_positive=rows),
              open('research/results/scenarios.json', 'w'), indent=1, ensure_ascii=False)

    rep = ['# Сценарный анализ (scenarios.py): сырой эдж vs комиссии\n']
    rep.append('Вариантов (символ×mask×idea×exit) с raw-IS+ и n>=25: %d. '
               'Из них raw-OOS+: %d. Выживают при maker-maker 4bp или vip 5bp: %d.\n' % (
                   len(rows), len(oos_raw_pos), len(surv)))
    rep.append('| symbol | mask | idea | exit | IS n/sumR(raw) | OOS n | OOS raw | OOS full | OOS mm4 | OOS vip |')
    rep.append('|---|---|---|---|---|---|---|---|---|---|')
    for r in oos_raw_pos[:20]:
        rep.append('| %s | %s | %s | %s | %d/%.1f | %d | %.1f | %.1f | %.1f | %.1f |' % (
            r['symbol'], r['mask'], r['idea'], r['exit'], r['is_n'], r['is_sum_R'],
            r['oos_n'], r['oos_raw_sumR'], r['oos_net']['full(m+t 7.5bp)'],
            r['oos_net']['maker-maker 4bp'], r['oos_net']['vip maker-t 5bp']))
    rep.append('\n## Выводы\n')
    if oos_raw_pos:
        rep.append('- Сырой (ZERO-fee) эдж существует у %d вариантов, лидер: **%s/%s/%s** (OOS raw %.1fR на %d сделках).\n' % (
            len(oos_raw_pos), oos_raw_pos[0]['symbol'], oos_raw_pos[0]['idea'],
            oos_raw_pos[0]['exit'], oos_raw_pos[0]['oos_raw_sumR'], oos_raw_pos[0]['oos_n']))
        if surv:
            rep.append('- При практике maker-maker (лимитный вход+выход) выживают %d вариантов — это путь: постOnly входы + лимитные TP/BE (уже в боте) есть, проверять на paper.\n' % len(surv))
        else:
            rep.append('- Но ни один не выживает даже при 4bp maker-maker: жёсткие издержки съедают весь raw-эдж. Нужен более крупный TF/дальние стопы.\n')
    else:
        rep.append('- Сырого эджа (без комиссий) тоже нет — сигнальная гипотеза целиком не работает.\n')
    with open('research/results/scenarios.md', 'w') as f:
        f.write('\n'.join(rep) + '\n')
    print('done. raw_is_pos=%d raw_oos_pos=%d surv=%d' % (
        len(rows), len(oos_raw_pos), len(surv)), flush=True)


if __name__ == '__main__':
    main()
