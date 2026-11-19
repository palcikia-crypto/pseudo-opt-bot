"""research/walkforward.py — walk-forward валидация: подбор параметров
на «тренировочном» окне, проверка на следующем за ним «тестовом» окне.
Исключает подглядывание в будущее: параметры выбираются только по прошлому.
"""
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.backtest import default_cfg, run

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
    if rep['n_trades'] < 8:
        return -9e9
    return rep['sum_R']


def main():
    from config import SYMBOLS
    from research.data import load_1m
    from research.backtest import SymbolData
    datas = [SymbolData(s, load_1m(s)) for s in SYMBOLS]
    bars = {d.symbol: d.bars for d in datas}
    t_start = min(int(b[0][0]) for b in bars.values())
    t_end = max(int(b[-1][0]) for b in bars.values())
    ndays = (t_end - t_start) / DAY
    n_folds = int(max(ndays - 4, 1))
    fold_len = DAY
    train_len = 3 * DAY
    print('days: %.1f, folds: %d' % (ndays, n_folds))

    results = []
    oos_trades = []
    for f in range(n_folds):
        tr0 = t_start + f * fold_len
        tr1 = tr0 + train_len
        te0, te1 = tr1, tr1 + fold_len
        if te0 >= t_end:
            break
        best, best_s = None, -1e18
        comb_rep = []
        for over in combos():
            cfg = _apply(default_cfg(), over)
            rep, _ = run(cfg, datas=datas, t0=tr0, t1=tr1)
            sc = _score(rep)
            comb_rep.append((over, rep['n_trades'], rep['sum_R'], sc))
            if sc > best_s:
                best_s, best = sc, over
        chosen = best if best is not None else {}
        cfg_test = _apply(default_cfg(), chosen)
        rep_oos, tr_oos = run(cfg_test, datas=datas, t0=te0, t1=te1)
        rep_base, tr_base = run(default_cfg(), datas=datas, t0=te0, t1=te1)
        for t in tr_oos:
            t['fold'] = f
            t['params'] = chosen
        oos_trades.extend(tr_oos)
        results.append(dict(
            fold=f,
            train=dict(t0=tr0, t1=tr1), test=dict(t0=te0, t1=te1),
            best_params=chosen, best_train_sum_R=round(best_s, 2),
            oos=dict(n=rep_oos['n_trades'], sum_R=rep_oos['sum_R'],
                     win_rate=rep_oos['win_rate'], net_pct=rep_oos['net_pnl_pct']),
            oos_default=dict(n=rep_base['n_trades'], sum_R=rep_base['sum_R'],
                             win_rate=rep_base['win_rate'], net_pct=rep_base['net_pnl_pct']),
            top3=sorted(comb_rep, key=lambda x: -x[3])[:3],
        ))
        print('fold %d: best=%s train_sum_R=%.2f -> OOS n=%d sum_R=%.2f (base: n=%d sum_R=%.2f)'
              % (f, json.dumps(chosen), best_s, rep_oos['n_trades'], rep_oos['sum_R'],
                 rep_base['n_trades'], rep_base['sum_R']))

    out = dict(n_folds=len(results), folds=results,
               oos_total=dict(
                   n_total=sum(f['oos']['n'] for f in results),
                   sum_R_total=round(sum(f['oos']['sum_R'] for f in results), 2),
                   sum_R_default=round(sum(f['oos_default']['sum_R'] for f in results), 2)))
    os.makedirs('research/results', exist_ok=True)
    with open('research/results/walkforward.json', 'w') as fp:
        json.dump(out, fp, indent=1, ensure_ascii=False)
    with open('research/results/walkforward_oos_trades.json', 'w') as fp:
        json.dump(oos_trades, fp, ensure_ascii=False)
    print('WALK-FORWARD OOS:', json.dumps(out['oos_total']))


if __name__ == '__main__':
    main()
