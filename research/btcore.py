"""research/btcore.py — универсальное ядро standalone-бек-тестера (раунд 9).

Слои (всё из реальных прогонов диалога, без дублирования логики):
  1. Данные: load_views() — 190д 1m × 15 монет + TF-проекции, ВСЕГДА выравнивание
     по timestamp (правило раунда 7); refresh_tail() — инкрементальное продление
     кэша свежими барами (не переписывая историю).
  2. Движок: lab.sim — эталон, не трогаем; Genome -> masks (lab/deep/micro2/union)
     -> run_window -> portfolio_accept -> stats/monthly.
  3. Комиссии: SCENARIOS (maker-taker full / mm4 / vip) через net_R.
  4. Проскальзывание: slippage_report() — live-стаканы loblog (спред bps,
     дисбаланс) -> реальная цена исполнения maker/taker.
  5. Валидация 1-в-1: shadow_report() — live-сделки paper_deep vs их бэктест-тень
     (расхождение R и причин выхода).
  6. Опцион-режим: option_edge() — годится ли сигнал под НАСТОЯЩИЙ опцион:
     движение до выхода против оценённой ATM-премии (σ·√t·k).

Запуск компонентов — см. evo.py (эволюция + daemon).
"""
import csv
import datetime
import glob
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import research.lab as lab
import research.deep as deep
import research.micro2 as micro2
from research.data import load_1m, _fetch_pages, cache_path

DAY = lab.DAY
LOB_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')


class Genome(dict):
    """Особь: symbol, tf (мин), idea, exit_name, params (маска+выход), gates."""

    @staticmethod
    def new(symbol, tf, idea, exit_name, params):
        return Genome(symbol=symbol, tf=tf, idea=idea, exit_name=exit_name,
                      params=params)

    @property
    def key(self):
        return '%s|%dm|%s|%s' % (self['symbol'], self['tf'], self['idea'],
                                 self['exit_name'])

    def to_json(self):
        return dict(self)

    @staticmethod
    def from_json(d):
        return Genome(d)

    def clone(self):
        return Genome(json.loads(json.dumps(dict(self))))


def load_views(days=190, symbols=None):
    from config import SYMBOLS
    full = [s.replace('/', '').split(':')[0] for s in (symbols or SYMBOLS)]
    raw = {sym: load_1m(sym, days=days) for sym in full}
    labs1 = {sym: lab.LabData(sym, b) for sym, b in raw.items()}
    for sym in full:
        if sym.startswith('BTC'):
            lab.BTC = labs1[sym]
    views = {1: labs1}
    for tf in (15, 60):
        views[tf] = {sym: deep.build_tf_lab(sym, raw[sym], tf) for sym in full}
    return full, raw, views


def masks_for(ld, genome):
    idea = genome['idea']
    cfg = dict(genome['params'])
    if idea.startswith('micro2:'):
        fams = micro2.micro2_families(ld, cfg)
        return fams[idea.split(':', 1)[1]]
    if idea.startswith('union:'):
        a, b = idea.split(':', 1)[1].split('+')
        allf = dict(lab.masks(ld, cfg))
        allf.update(micro2.micro2_families(ld, cfg))
        return (np.logical_or(allf[a][0], allf[b][0]),
                np.logical_or(allf[a][1], allf[b][1]))
    return lab.masks(ld, cfg)[idea]


def run_genome(views, genome, lo, hi):
    ld = views[genome['tf']][genome['symbol']]
    try:
        larr, sarr = masks_for(ld, genome)
    except KeyError:
        return []
    ex = dict(genome['params']['exit'])
    trades = []
    mask = np.flatnonzero((larr | sarr) & (ld.ts >= lo) & (ld.ts < hi))
    next_free = -1
    for i in mask:
        if i <= next_free:
            continue
        side = 'long' if larr[i] else 'short'
        t = lab.sim(ld, ex, int(i), side, hi)
        if t is None:
            continue
        next_free = int(i) + t['bars_held']
        trades.append(t)
    return trades


def evaluate(views, genome, lo, hi, apply_portfolio=True):
    trades = run_genome(views, genome, lo, hi)
    if apply_portfolio:
        trades, _ = lab.portfolio_accept(trades, genome['params']['exit'], hi)
    st = lab.stats(trades)
    mon = lab.monthly_table(trades)
    months = len(mon)
    mpos = sum(1 for v in mon.values() if v['sum_R'] > 0)
    st['months'] = months
    st['months_pos'] = mpos
    st['fitness'] = (st['sum_R'] * (mpos / months)) if months and st['n'] >= 10 else -999.0
    return st, trades


def refresh_tail(symbol, tail_days=3):
    """Дописать свежие бары в кэш, не трогая историю (инкремент для демона)."""
    path = cache_path(symbol)
    old = load_1m(symbol)
    if not old:
        return 0
    t_last = old[-1][0]
    fresh = _fetch_pages(symbol, int(__import__('time').time() * 1000) + 60000,
                         tail_days * 1440 + 60)
    merged = dict((r[0], r) for r in old)
    for r in fresh:
        merged[int(r[0])] = (int(r[0]), float(r[1]), float(r[2]), float(r[3]),
                             float(r[4]), float(r[5]))
    rows = [merged[k] for k in sorted(merged)]
    added = sum(1 for r in rows if r[0] > t_last)
    tmp = path + '.tmp'
    with gzip.open(tmp, 'wt', newline='') as f:
        csv.writer(f).writerows([[r[0], r[1], r[2], r[3], r[4], r[5]] for r in rows])
    os.replace(tmp, path)
    return added


def slippage_report(max_rows=200000):
    """Live-стаканы loblog: медианный спред, средний дисбаланс — реальная цена исполнения.
    Свежий файл может быть mid-write (EOFError) — идём от новых к старым."""
    out = {}
    byday = sorted(glob.glob(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'cache', 'lob_*.csv.gz')),
        reverse=True)
    for path in byday:
        sym = os.path.basename(path).split('_')[1]
        if sym in out or sym == '':
            continue
        try:
            with gzip.open(path, 'rt') as f:
                rows = list(csv.reader(f))[-max_rows:]
            if len(rows) < 100:
                continue
            hdr = rows[0]
            si = hdr.index('spread_bps')
            im = hdr.index('imb25') if 'imb25' in hdr else None
            vals = [float(r[si]) for r in rows[1:] if r[si]]
            imbs = [float(r[im]) for r in rows[1:] if r[im]] if im else []
            out[sym] = dict(median_spread_bps=round(float(np.median(vals)), 2),
                            p90_spread_bps=round(float(np.percentile(vals, 90)), 2),
                            mean_imb25=round(float(np.mean(imbs)), 3) if imbs else None,
                            n=int(len(vals)))
        except Exception:
            continue
    return out


def shadow_report(path=None):
    """Валидация 1-в-1: live paper_deep vs бэктест-тень той же сделки."""
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'results', 'paper_deep.json')
    if not os.path.exists(path):
        return dict(trades=0, note='нет live-сделок')
    st = json.load(open(path))
    diffs, matched = [], 0
    for t in st.get('trades', []):
        sh = t.get('shadow')
        if not sh:
            continue
        diffs.append(abs(t['R'] - sh['R']))
        if sh.get('exit_reason') == t.get('exit_reason'):
            matched += 1
    if not diffs:
        return dict(trades=len(st.get('trades', [])), matched=0,
                    mean_abs_R_diff=None)
    return dict(trades=len(diffs), matched=matched,
                match_rate_pct=round(100.0 * matched / len(diffs), 1),
                mean_abs_R_diff=round(float(np.mean(diffs)), 3),
                max_abs_R_diff=round(float(np.max(diffs)), 3))


def option_edge(trades, hold_min_med):
    """Годится ли сигнал под настоящий опционы: средняя рыночная ATM-премия
    (≈0.8·σ_1m·√hold) против реального движения сделки (R·sl_pct).
    edge_bp > 0 == премия отрабатывается в плюс."""
    if not trades:
        return None
    moves, prems = [], []
    for t in trades:
        mv = abs(t['R']) * t['sl_pct']
        prem = 0.8 * (t['sl_pct'] / 1.5) * float(np.sqrt(max(hold_min_med, 1.0)))
        moves.append(mv)
        prems.append(prem)
    edge = float(np.mean(moves) - np.mean(prems))
    return dict(premium_bp=round(float(np.mean(prems)) * 1e4, 1),
                move_bp=round(float(np.mean(moves)) * 1e4, 1),
                edge_bp=round(edge * 1e4, 1),
                verdict='option-viable' if edge > 0 else 'no-for-options')
