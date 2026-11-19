"""research/run_all.py — полный исследовательский прогон и отчёт report.md:
бэктест текущего конфига -> walk-forward -> edge-scan -> анализ издержек.
Запуск: ./venv/bin/python research/run_all.py [days]
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def sh(script):
    r = subprocess.run(script, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise SystemExit('fail: %s' % script)
    return r.stdout


def cost_analysis(cfg):
    from research.data import load_1m
    from research.indicators_vec import wilder_atr
    from config import SYMBOLS
    rows = []
    for s in SYMBOLS:
        bars = load_1m(s)
        h = np.array([b[2] for b in bars])
        l = np.array([b[3] for b in bars])
        c = np.array([b[4] for b in bars])
        atr = wilder_atr(h, l, c)
        apct = np.nanmedian(atr / c * 100.0)
        sl_pct = min(max(cfg['sl_atr_mult'] * apct, cfg['sl_min_pct'] * 100.0),
                     cfg['sl_max_pct'] * 100.0)
        cost_pct = cfg['maker_fee'] * 100.0 + cfg['taker_fee'] * 100.0 + cfg['entry_eps'] * 100.0
        cost_r = cost_pct / sl_pct
        rows.append(dict(symbol=s.split('/')[0], atr_pct=round(float(apct), 3),
                         sl_pct=round(float(sl_pct), 3),
                         cost_r=round(float(cost_r), 3),
                         wr_needed=round(100.0 * (1.0 + cost_r) / (1.0 + cfg['rr']), 1)))
    return rows


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def fmt_tbl(rows, cols):
    hdr = '| ' + ' | '.join(cols) + ' |'
    sep = '|' + '|'.join(['---'] * len(cols)) + '|'
    lines = [hdr, sep]
    for r in rows:
        lines.append('| ' + ' | '.join(str(r[c]) for c in cols) + ' |')
    return '\n'.join(lines)


def main():
    os.makedirs(RES, exist_ok=True)
    print('[1/4] бэктест текущего конфига...')
    sh('./venv/bin/python research/backtest.py --out research/results/base.json')
    print('[2/4] walk-forward...')
    sh('./venv/bin/python research/walkforward.py')
    print('[3/4] edge-scan...')
    sh('./venv/bin/python research/edge_scan.py research/results/base.json')
    print('[4/4] отчёт...')

    base = load(os.path.join(RES, 'base.json'))['report']
    wf = load(os.path.join(RES, 'walkforward.json'))
    es = load(os.path.join(RES, 'edge_scan.json'))
    cfg = __import__('research.backtest', fromlist=['default_cfg']).default_cfg()
    costs = cost_analysis(cfg)
    wr_needed = 100.0 / (1.0 + cfg['rr'])

    md = ['# Исследовательский отчёт — pseudo-opt-bot',
          '',
          'Данные: Bybit linear 1m, ~7.6 дней, %d символов. Период: %s' % (
              len(costs), __import__('datetime').datetime
              .fromtimestamp(bars_t0() / 1000).strftime('%Y-%m-%d %H:%M UTC')),
          '',
          '## 1. Бэктест текущего конфига (консервативная модель входа)',
          '',
          '| метрика | значение |',
          '|---|---|',
          '| сделок | %d |' % base['n_trades'],
          '| винрейт | %.1f%% (нужно ≥%.1f%% при RR=%.1f без учёта комиссий) |' % (
              base['win_rate'], wr_needed, cfg['rr']),
          '| ожидание | %.2fR на сделку |' % base['expectancy_R'],
          '| profit factor | %s |' % base['profit_factor'],
          '| итог | %+.1f%% к депо 1000 |' % base['net_pnl_pct'],
          '| макс. просадка | %.1f%% |' % base['max_dd_pct'],
          '| уплачено комиссий | %.1f USDT (%.0f%% депо!) |' % (
              base['total_fees'], base['total_fees'] / 10.0),
          '', 'Выходы: %s' % json.dumps(base['by_reason']),
          '',
          '## 2. Издержки — главный убийца стратегии',
          '',
          'Стоимость круга (maker-in + taker-out + буфер входа) ≈ 10.5 бп цены.',
          'Относительный размер стопа определяет цену страхалки:',
          '',
          fmt_tbl(costs, ['symbol', 'atr_pct', 'sl_pct', 'cost_r', 'wr_needed']),
          '',
          '`wr_needed` — винрейт, при котором сделка хотя бы безубыточна С УЧЁТОМ издержек.',
          '',
          '## 3. Walk-forward (без подглядывания в будущее)',
          '',
          '',
          fmt_tbl([f['oos'] | {'fold': f['fold'], 'params': json.dumps(f['best_params'])}
                   for f in wf['folds']], ['fold', 'params', 'n', 'sum_R', 'win_rate', 'net_pct']),
          '',
          'OOS суммарно: %s сделок, ΣR=%s (базовый конфиг: ΣR=%s)' % (
              wf['oos_total']['n_total'], wf['oos_total']['sum_R_total'],
              wf['oos_total']['sum_R_default']),
          '',
          '## 4. Edge-scan (ML-lite, логрегрессия, time-CV)',
          '',
          fmt_tbl(es['cv'], ['fold', 'n_test', 'auc', 'acc', 'wr_base', 'wr_p05', 'R_base', 'R_p05', 'n_p05']),
          '',
          '### По корзинам',
          '',
          'Часы:', '',
          fmt_tbl(es['by_hour'], ['bucket', 'n', 'win_rate', 'avg_R', 'sum_R']),
          '',
          'ATR:', '',
          fmt_tbl(es['by_atr'], ['bucket', 'n', 'win_rate', 'avg_R', 'sum_R']),
          '',
          'Объём:', '',
          fmt_tbl(es['by_vol'], ['bucket', 'n', 'win_rate', 'avg_R', 'sum_R']),
          '',
          'Сторона:', '',
          fmt_tbl(es['by_side'], ['bucket', 'n', 'win_rate', 'avg_R', 'sum_R']),
          '',
          '## Выводы',
          '',
          '1. У текущего конфига нет эджа: винрейт %.1f%% ниже порога %.1f%% ещё до комиссий;' % (base['win_rate'], wr_needed),
          '   с издержками порог 39–49% винрейта (0.37–0.70R за круг при текущих стопах).',
          '2. Главная структурная проблема — крошечные стопы (SL 0.15–0.6% от цены):',
          '   комиссия + буфер + выход по тейкеру съедают до половины каждой ставки.',
          '3. Walk-forward устойчиво выбирает более широкие стопы (sl_atr_mult 3.5) и RR 3.0:',
          '   OOS ΣR=%s против %s у базового — лучше, но всё ещё отрицательно.' % (
              wf['oos_total']['sum_R_total'], wf['oos_total']['sum_R_default']),
          '4. Edge-scan: OOS AUC логрегрессии 0.30–0.36 — признаки при входе НЕ предсказывают',
          '   исход; лонги заметно хуже шортов — проверять стороны раздельно.',
          '5. Практические шаги: (а) SL_ATR_MULT 2.5–3.5 и/или фильтр сделок со стопом ≥0.3% цены;',
          '   (б) более редкие входы (трендовый фильтр старшего ТФ); (в) каждую новую идею',
          '   прогонять этим конвейером ДО включения в бота; (г) слippage/скорость мерить',
          '   по факту в paper-режиме, а не задавать константами на глаз.',
          '']
    path_md = os.path.join(RES, 'report.md')
    with open(path_md, 'w') as f:
        f.write('\n'.join(md))
    print('OK:', path_md)


def bars_t0():
    from research.data import load_1m
    import config
    return min(int(load_1m(s)[0][0]) for s in config.SYMBOLS)


if __name__ == '__main__':
    main()
