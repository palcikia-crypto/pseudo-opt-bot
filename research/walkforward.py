"""research/walkforward.py — walk-forward валидация на длинной истории:
скользящие окна (train -> test), выбор параметров ТОЛЬКО по прошлому;
в конце — consensus-параметры и помесячная устойчивость на всей истории.
"""
import collections
import datetime
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.backtest import SymbolData, default_cfg, run

DAY = 86400000

GRID = {
    'sl_atr_mult': [1.5, 2.5, 3.5],
    'rr': [2.0, 2.5, 3.0],
    'vol_mult': [1.3, 1.8],
}


def combos():
    keys = list(GRID)
    for vals in itertools.product(*GRID.values()):
        yield dict(zip(keys, vals))


def _apply(base, over):
    c = dict(base)
    c.update(over)
    return c


def _score(rep):
    if rep['n_trades'] < 10:
        return -9e9
    return rep['sum_R']


def month_key(ts):
    return datetime.datetime.fromtimestamp(ts / 1000, datetime.timezone.utc).strftime('%Y-%m')


def main():
    from research.data import load_1m
    from config import SYMBOLS

    datas = [SymbolData(s, load_1m(s)) for s in SYMBOLS]
    t_start = min(int(d.ts[0]) for d in datas)
    t_end = max(int(d.ts[-1]) for d in datas)
    span_days = (t_end - t_start) / DAY
    n_folds = max(1, int((span_days - 90) // 30))
    if span_days < 40:
        train_days, test_days = span_days * 0.6, span_days * 0.4
        n_folds = 1
    else:
        train_days, test_days = 90.0, 30.0
        if t_start + (n_folds * test_days + train_days) * DAY > t_end:
            n_folds = max(1, int((span_days - train_days) / test_days))
    print('span %.0f days, folds %d, train %.0f, test %.0f' %
          (span_days, n_folds, train_days, test_days))

    results, oos_trades = [], []
    wins_selected = collections.Counter()
    for f in range(n_folds):
        tr0 = t_start + f * test_days * DAY
        tr1 = tr0 + train_days * DAY
        te0, te1 = tr1, tr1 + test_days * DAY
        if te0 >= t_end:
            break
        best, best_s = None, -1e18
        table = []
        for over in combos():
            rep, _ = run(_apply(default_cfg(), over), datas=datas, t0=tr0, t1=tr1)
            sc = _score(rep)
            table.append((over, rep['n_trades'], rep['sum_R'], sc))
            if sc > best_s:
                best_s, best = sc, over
        chosen = best if best is not None else {}
        cfg_over = chosen or dict(vol_mult=1.8)
        wins_selected[json.dumps(cfg_over, sort_keys=True)] += 1
        rep_oos, tr_oos = run(_apply(default_cfg(), cfg_over), datas=datas, t0=te0, t1=te1)
        rep_base, _ = run(default_cfg(), datas=datas, t0=te0, t1=te1)
        for t in tr_oos:
            t['fold'] = f
            t['params'] = cfg_over
        oos_trades.extend(tr_oos)
        results.append(dict(
            fold=f, train_days=round((tr1 - tr0) / DAY), test_days=round((te1 - te0) / DAY),
            best_params=cfg_over, best_train_sum_R=round(best_s, 2),
            oos=dict(n=rep_oos['n_trades'], sum_R=rep_oos['sum_R'],
                     win_rate=rep_oos['win_rate'], net_pct=rep_oos['net_pnl_pct']),
            oos_default=dict(n=rep_base['n_trades'], sum_R=rep_base['sum_R'],
                             win_rate=rep_base['win_rate'], net_pct=rep_base['net_pnl_pct']),
            top3=[(t[0], t[1], t[2]) for t in sorted(table, key=lambda x: -x[3])[:3]],
        ))
        print('fold %d: %s train_sumR=%.2f -> OOS n=%d sumR=%.2f (base %.2f)' %
              (f, json.dumps(cfg_over), best_s, rep_oos['n_trades'],
               rep_oos['sum_R'], rep_base['sum_R']))

    consensus_src = wins_selected.most_common(1)[0][0] if wins_selected else '{}'
    consensus = json.loads(consensus_src)
    rep_cons, tr_cons = run(_apply(default_cfg(), consensus), datas=datas)
    months = collections.defaultdict(lambda: dict(n=0, sum_R=0.0, wins=0))
    for t in tr_cons:
        m = months[month_key(t['ts_entry'])]
        m['n'] += 1
        m['sum_R'] += t['R']
        m['wins'] += 1 if t['pnl'] > 0 else 0
    monthly = [dict(month=k, n=v['n'], sum_R=round(v['sum_R'], 2),
                    win_rate=round(100.0 * v['wins'] / v['n'], 1))
               for k, v in sorted(months.items())]
    pos_months = sum(1 for m in monthly if m['sum_R'] > 0)

    out = dict(
        span_days=round(span_days), n_folds=len(results),
        consensus=consensus,
        consensus_full=dict(n=rep_cons['n_trades'], sum_R=rep_cons['sum_R'],
                            win_rate=rep_cons['win_rate'],
                            net_pct=rep_cons['net_pnl_pct'],
                            max_dd_pct=rep_cons['max_dd_pct'],
                            profit_factor=rep_cons['profit_factor'],
                            expectancy_R=rep_cons['expectancy_R'],
                            months_pos='%d/%d' % (pos_months, len(monthly))),
        monthly=monthly,
        oos_total=dict(
            n_total=sum(f['oos']['n'] for f in results),
            sum_R_total=round(sum(f['oos']['sum_R'] for f in results), 2),
            sum_R_default=round(sum(f['oos_default']['sum_R'] for f in results), 2)),
        folds=results,
    )
    os.makedirs('research/results', exist_ok=True)
    with open('research/results/walkforward.json', 'w') as fp:
        json.dump(out, fp, indent=1, ensure_ascii=False)
    with open('research/results/walkforward_oos_trades.json', 'w') as fp:
        json.dump(oos_trades, fp, ensure_ascii=False)
    print('CONSENSUS', consensus, out['consensus_full'])
    print('OOS', out['oos_total'])


if __name__ == '__main__':
    main()
