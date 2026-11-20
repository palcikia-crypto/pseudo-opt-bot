"""research/optcore.py — ОПЦИОН-РЕЖИМ бэк-тестера (раунд 9b).

Сигнал из любого семейства (lab/micro2/union) превращается в покупку
НАСТОЯЩЕГО опциона:
  - вход: open следующего бара TF (как в live), направление = сигнал;
  - экспирация: фиксированная E минут (геном), держим до конца — European;
  - премия ATM: prem_k × σ_1m × √E  (σ_1m = std 1m-доходностей символа —
    реальная realized vol, не ATR-прикидка);
  - ванильный контраст: payoff = max(0, dir·(close_expiry/entry − 1)) − премия;
  - цифровой (binary): win (+0.85 × ставка) / loss (−1 × ставка).

Фитнес = сумма ванильной выплаты (bp от номинала) × доля месяцев в плюс,
гейт n≥20. Всё «один-в-один» с той же машиной данных btcore/lab.
Automation: optevo.py (эволюция + демон).
"""
import datetime
import json
import os

import numpy as np

import research.btcore as bt

MAX_HOLD_BARS_CAP = 20000


def sigma_1m(lab1):
    """Реализованная 1m-волатильность (фракция за минуту) символа."""
    r = np.asarray(lab1.ret1, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 1000:
        r = r[np.isfinite(r)] if r.size else np.array([np.nan])
    return float(np.std(r))


def option_trades(views, genome, lo, hi):
    """Симуляция опционных сделок по сигналам генома в окне [lo, hi)."""
    tf = int(genome['tf'])
    sym = genome['symbol']
    ld = views[tf][sym]
    sigma = sigma_1m(views[1][sym])
    p = genome['params']
    prem_k = float(p.get('prem_k', 0.8))
    E = int(p.get('expiry_min', 60))
    try:
        larr, sarr = bt.masks_for(ld, genome)
    except KeyError:
        return []
    idx = np.flatnonzero((larr | sarr) & (ld.ts >= lo) & (ld.ts < hi))
    e_bars = max(1, E // tf)
    prem = prem_k * sigma * float(np.sqrt(E))
    out = []
    next_free = -1
    n = ld.ts.size
    for i in idx:
        i = int(i)
        if i <= next_free or i + 1 >= n:
            continue
        j = i + 1 + e_bars
        if j >= n:
            continue
        side = 'long' if larr[i] else 'short'
        entry = float(ld.o[i + 1])
        if not np.isfinite(entry) or entry <= 0:
            continue
        mv = (float(ld.c[j]) / entry - 1.0) * (1.0 if side == 'long' else -1.0)
        vanilla = max(0.0, mv) - prem
        out.append(dict(symbol=sym, side=side, minute=i,
                     ts_entry=int(ld.ts[i + 1]), ts_expiry=int(ld.ts[j]),
                     expiry_min=E, entry=entry, exit_px=float(ld.c[j]),
                     move_pct=round(mv * 1e4, 2),
                     prem_pct=round(prem * 1e4, 2),
                     vanilla_bp=round(vanilla * 1e4, 2),
                     digital_win=bool(mv > 0),
                     R=round(vanilla / prem, 3) if prem > 0 else 0.0, tf=tf))
        next_free = i + e_bars
    return out


def evaluate_option(views, genome, lo, hi, apply_portfolio=True):
    tr = option_trades(views, genome, lo, hi)
    if not tr:
        return dict(n=0, sum_bp=0.0, sum_R_prem=0.0, wr_vanilla=0.0,
                    wr_digital=0.0, expectancy_bp=0.0, months=0, months_pos=0,
                    fitness=-999.0), []
    vbp = np.array([t['vanilla_bp'] for t in tr])
    dig = np.array([1.0 if t['digital_win'] else 0.0 for t in tr])
    prem_bp = float(tr[0]['prem_pct'])
    dnet = dig.sum() * 0.85 - (len(tr) - dig.sum())
    mon = {}
    for t, w in zip(tr, dig):
        key = datetime.datetime.fromtimestamp(t['ts_entry'] / 1000,
                                              datetime.timezone.utc).strftime('%Y-%m')
        mon.setdefault(key, []).append(0.85 if w > 0 else -1.0)
    months = len(mon)
    mpos = sum(1 for v in mon.values() if sum(v) > 0)
    sum_bp = float(vbp.sum())
    fitness = (dnet * (mpos / months)) if months and len(tr) >= 20 else -999.0
    st = dict(n=len(tr), sum_bp=round(sum_bp, 1),
              sum_R_prem=round(dnet, 1),
              prem_bp=round(prem_bp, 2),
              wr_vanilla=round(100.0 * float((vbp > 0).mean()), 1),
              wr_digital=round(100.0 * float(dig.mean()), 1),
              expectancy_bp=round(float(vbp.mean()), 2),
              months=months, months_pos=mpos, fitness=fitness)
    return st, tr


def expiry_scan(views, genome_tmpl, expiries=(20, 30, 60, 120, 240, 480)):
    """Быстрый скан экспираций для одного генома (обезреженный выбор)."""
    rows = []
    for E in expiries:
        g = genome_tmpl.clone()
        g['params'] = dict(g['params'], expiry_min=E)
        st, _ = evaluate_option(views, g, 0, 9 ** 12)
        rows.append((E, st['n'], st['sum_bp'], st['wr_vanilla'], st['wr_digital']))
    return rows


def opt_snapshot_summary(st):
    return '%+.1fbp/n=%d | dig %+.1fR' % (st['sum_bp'], st['n'], st['sum_R_prem'])
