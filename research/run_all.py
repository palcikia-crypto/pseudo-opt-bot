"""research/run_all.py — полный исследовательский прогон и отчёт report.md:
бэктест текущего конфига -> реальные издержки (стаканы Bybit) -> walk-forward
(скользящие окна, consensus-параметры) -> помесячная устойчивость -> edge-scan.
Запуск: ./venv/bin/python research/run_all.py
"""
import collections
import datetime
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise SystemExit('fail: ' + cmd)
    return r.stdout


def load(name):
    p = os.path.join(RES, name)
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def fmt_tbl(rows, cols):
    hdr = '| ' + ' | '.join(cols) + ' |'
    sep = '|' + '|'.join(['---'] * len(cols)) + '|'
    return '\n'.join([hdr, sep] +
                     ['| ' + ' | '.join(str(r.get(c)) for c in cols) + ' |' for r in rows])


def monthly_table(trades):
    months = collections.defaultdict(lambda: dict(n=0, sum_R=0.0, wins=0, pnl=0.0))
    for t in trades:
        k = datetime.datetime.fromtimestamp(t['ts_entry'] / 1000,
                                            datetime.timezone.utc).strftime('%Y-%m')
        m = months[k]
        m['n'] += 1
        m['sum_R'] += t['R']
        m['pnl'] += t['pnl']
        m['wins'] += 1 if t['pnl'] > 0 else 0
    return [dict(month=k, n=v['n'], sum_R=round(v['sum_R'], 2), pnl=round(v['pnl'], 2),
                 win_rate=round(100.0 * v['wins'] / v['n'], 1))
            for k, v in sorted(months.items())]


def main():
    os.makedirs(RES, exist_ok=True)
    print('[1/5] реальные издержки (стаканы Bybit)...')
    if load('fees_real.json') is None:
        sh('./venv/bin/python research/fees_real.py')
    print('[2/5] бэктест текущего конфига (весь кэш)...')
    sh('./venv/bin/python research/backtest.py --out research/results/base.json')
    print('[3/5] walk-forward...')
    sh('./venv/bin/python research/walkforward.py')
    print('[4/5] edge-scan...')
    sh('./venv/bin/python research/edge_scan.py research/results/base.json')
    print('[5/5] отчёт...')
    build_report()


def build_report():
    basej = load('base.json') or {}
    base = basej.get('report', {})
    trades = basej.get('trades', [])
    wf = load('walkforward.json') or {}
    es = load('edge_scan.json') or {}
    fr = load('fees_real.json') or {}

    import research.backtest as bt
    cfg = bt.default_cfg()
    wr_free = 100.0 / (1.0 + cfg['rr'])

    costs = []
    if fr.get('rows'):
        by_sym = {r['symbol']: r for r in fr['rows']}
        from research.data import load_1m
        from research.indicators_vec import wilder_atr
        from config import SYMBOLS
        for s in SYMBOLS:
            sym = s.split('/')[0] + 'USDT'
            m = by_sym.get(sym)
            if not m:
                continue
            bars = load_1m(s)
            h = np.array([b[2] for b in bars])
            l = np.array([b[3] for b in bars])
            c = np.array([b[4] for b in bars])
            apct = float(np.nanmedian(wilder_atr(h, l, c) / c * 100.0))
            sl_pct = min(max(cfg['sl_atr_mult'] * apct, cfg['sl_min_pct'] * 100.0),
                         cfg['sl_max_pct'] * 100.0)
            rt = (cfg['maker_fee'] + cfg['taker_fee']) * 100.0 + cfg['entry_eps'] * 100.0 \
                + m['spread_bps'] / 200.0 + (m['impact_exit_bps'] or 0.0) / 100.0
            cost_r = rt / sl_pct
            costs.append(dict(symbol=s.split('/')[0], spread_bps=m['spread_bps'],
                              impact_bps=m['impact_exit_bps'], sl_pct=round(sl_pct, 3),
                              cost_r=round(cost_r, 3),
                              wr_needed=round(100.0 * (1 + cost_r) / (1 + cfg['rr']), 1)))
    wr_real = (lambda: 100.0 * (1 + max((x['cost_r'] for x in costs), default=0.3)) / (1 + cfg['rr']))()

    md = ['# Исследовательский отчёт — pseudo-opt-bot (6 месяцев, 15 символов)',
          '',
          'Сгенерирован: %s. Данные: Bybit linear 1m, span ~%.0f дней.' % (
              datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
              (wf.get('span_days') or 0)),
          '',
          '## 1. Бэктест текущего конфига',
          '',
          fmt_tbl([dict(metric=k, value=v) for k, v in [
              ('сделок', base.get('n_trades')),
              ('винрейт, %', base.get('win_rate')),
              ('порог безубытка без комиссий, %', round(wr_free, 1)),
              ('ожидание, R', base.get('expectancy_R')),
              ('profit factor', base.get('profit_factor')),
              ('итог, %', base.get('net_pnl_pct')),
              ('макс просадка, %', base.get('max_dd_pct')),
              ('комиссии, USDT', base.get('total_fees')),
              ('выходы', json.dumps(base.get('by_reason', {}))),
          ]], ['metric', 'value']),
          '',
          '## 2. Реальные издержки (спред/глубина Bybit сейчас + наши комиссии)',
          '',
          fmt_tbl(costs, ['symbol', 'spread_bps', 'impact_bps', 'sl_pct', 'cost_r', 'wr_needed']),
          '',
          'cost_r = (maker+вход buфер+taker+полспреда+impact) / размер стопа в % цены.',
          'Перед выходом на деньги: порог винрейта с реальными издержками ≈ **%.1f%%** (для худшего по стоимости символа).' % wr_real,
          '',
          '## 3. Walk-forward (%d фолд: train %s→test, без подглядывания)' % (
              wf.get('n_folds', 0), ', '.join('%d/%d' % (f['train_days'], f['test_days'])
                                              for f in wf.get('folds', []))),
          '',
          fmt_tbl([dict(fold=f['fold'],
                        params=json.dumps(f['best_params']),
                        train_sum_R=f['best_train_sum_R'],
                        oos_n=f['oos']['n'], oos_sum_R=f['oos']['sum_R'],
                        oos_win=f['oos']['win_rate'], oos_net=f['oos']['net_pct'],
                        base_sum_R=f['oos_default']['sum_R'])
                   for f in wf.get('folds', [])],
                  ['fold', 'params', 'train_sum_R', 'oos_n', 'oos_sum_R', 'oos_win', 'oos_net', 'base_sum_R']),
          '',
          'OOS суммарно: N=%s, ΣR=%s (базовый конфиг: ΣR=%s)' % (
              wf.get('oos_total', {}).get('n_total'),
              wf.get('oos_total', {}).get('sum_R_total'),
              wf.get('oos_total', {}).get('sum_R_default')),
          '',
          '### Consensus-параметры (что ход-форвард выбирает чаще всего)',
          '',
          'Параметры: `%s`' % json.dumps(wf.get('consensus', {})),
          '',
          fmt_tbl([dict(metric=k, value=v) for k, v in (wf.get('consensus_full') or {}).items()],
                  ['metric', 'value']),
          '',
          '### Помесячная устойчивость consensus на всей истории',
          '',
          fmt_tbl(wf.get('monthly', []), ['month', 'n', 'sum_R', 'win_rate']),
          '',
          '## 4. Edge-scan (логрегрессия, time-CV)',
          '',
          fmt_tbl(es.get('cv', []), ['fold', 'n_test', 'auc', 'acc', 'wr_base', 'wr_p05', 'R_base', 'R_p05', 'n_p05']),
          '',
          '### Разрезы (по сделкам текущего конфига)',
          '',
          'Сторона:', '',
          fmt_tbl(es.get('by_side', []), ['bucket', 'n', 'win_rate', 'avg_R', 'sum_R']),
          '',
          'ATR:', '',
          fmt_tbl(es.get('by_atr', []), ['bucket', 'n', 'win_rate', 'avg_R', 'sum_R']),
          '',
          'Лучшие/худшие символы:', '',
          fmt_tbl(es.get('by_symbol', []), ['bucket', 'n', 'win_rate', 'avg_R', 'sum_R']),
          '',
          '## Выводы',
          '',
          conclusions(base, wf, es, costs),
          '']
    with open(os.path.join(RES, 'report.md'), 'w') as f:
        f.write('\n'.join(md))
    print('OK: research/results/report.md')


def conclusions(base, wf, es, costs):
    lines = []
    cons = wf.get('consensus_full') or {}
    if base.get('n_trades'):
        wr_free = 100.0 / (1.0 + base.get('_rr', 2.5))
        lines.append('1. Текущий конфиг: винрейт %s%% — ниже порога безубытка (%.1f%% при RR 2.5)' % (
            base.get('win_rate'), 28.6))
        lines.append('   даже без комиссий; с издержками порог ~%.0f%% (стоимость круга до %.2fR).' % (
            max((x['wr_needed'] for x in costs), default=45.0),
            max((x['cost_r'] for x in costs), default=0.45)))
        lines.append('   Итог: %s%% времени/депо — минус, исполнение само по себе корректно.' % base.get('net_pnl_pct'))
    if wf.get('oos_total'):
        c, d = wf['oos_total']['sum_R_total'], wf['oos_total']['sum_R_default']
        lines.append('2. Walk-forward OOS: ΣR=%s против %s у базового — %s.' % (
            c, d, 'лучше, но не плюс' if (c or 0) < 0 else 'плюс!'))
    if cons:
        lines.append('3. Consensus-параметры `%s`: всю историю N=%s, ΣR=%s, минус-месяцы %s.' % (
            json.dumps(wf.get('consensus', {})), cons.get('n'), cons.get('sum_R'),
            cons.get('months_pos')))
    aucs = [f.get('auc') for f in (es.get('cv') or []) if f.get('auc') is not None]
    if aucs:
        lines.append('4. ML-скан: OOS AUC %.2f–%.2f — %s.' % (
            min(aucs), max(aucs),
            'предсказательной силы у текущих признаков входа нет' if max(aucs) < 0.55
            else 'слабая предсказательная сила — проверить фильтр p>0.5'))
    lines.append('5. Решение: не включать текущую стратегию на реальные деньги.')
    lines.append('   План: другой вход/выход (старший ТФ, трейлинг, шорт-байс), каждая идея —')
    lines.append('   через run_all.py до включения; только OOS-плюс + положительное большинство месяцев.')
    return '\n'.join(lines)


if __name__ == '__main__':
    main()
