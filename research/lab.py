"""research/lab.py — лаборатория поиска эджа: перебор семейств сигналов
и типов выходов на 6-месячной 1m-истории с консервативной моделью исполнения.

Протокол: IS = первые 120 дней, OOS = последние 60 дней. Выбор идей — ТОЛЬКО
по IS; OOS — чистая проверка. Затем помесячная устойчивость победителей.

Запуск: ./venv/bin/python research/lab.py [--quick] [--symbols BTCUSDT,SOLUSDT]
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.backtest import _sl_dist  # noqa: F401  (общая модель)
from research.data import load_1m
from research.indicators_vec import (ema_series, rollmax_prev, rollmin_prev,
                                     rolling_mean, rolling_std, wilder_atr,
                                     wilder_rsi)

DAY = 86400000
MAKER = 0.0002
TAKER = 0.00055
EPS = 0.0003


def to_tf(rows, minutes):
    """Агрегация строк ohlcv к TF; минуты кратны 5."""
    out = []
    bucket = None
    ms = minutes * 60000
    for r in rows:
        b = r[0] // ms
        if bucket is None or bucket[0] != b:
            if bucket is not None:
                out.append(bucket[1])
            bucket = (b, [b * ms, r[1], r[2], r[3], r[4], r[5]])
        else:
            q = bucket[1]
            q[2] = max(q[2], r[2])
            q[3] = min(q[3], r[3])
            q[4] = r[4]
            q[5] += r[5]
    if bucket is not None:
        out.append(bucket[1])
    return out


class LabData:
    """Все признаки по символу, посчитанные один раз (numpy)."""

    def __init__(self, symbol, bars):
        self.symbol = symbol
        n = len(bars)
        self.ts = np.array([b[0] for b in bars], dtype=np.int64)
        self.o = np.array([b[1] for b in bars], dtype=float)
        self.h = np.array([b[2] for b in bars], dtype=float)
        self.l = np.array([b[3] for b in bars], dtype=float)
        self.c = np.array([b[4] for b in bars], dtype=float)
        self.v = np.array([b[5] for b in bars], dtype=float)
        self.atr = wilder_atr(self.h, self.l, self.c)
        self.atr_pct = self.atr / self.c * 100.0
        self.rsi14 = wilder_rsi(self.c, 14)
        self.rsi5 = wilder_rsi(self.c, 5)
        self.vol_base = rolling_mean(self.v, 20, shift=1)
        self.vol_ratio = self.v / np.where(self.vol_base > 0, self.vol_base, np.nan)
        mid = rolling_mean(self.c, 20)
        sd = rolling_std(self.c, 20)
        self.bb_up = mid + 2.0 * sd
        self.bb_lo = mid - 2.0 * sd
        width = 4.0 * sd
        sq = np.full(n, np.nan)
        finite_w = np.isfinite(width)
        w0 = int(np.argmax(finite_w))
        if finite_w[w0]:
            nw = width[w0:]
            wm = rolling_mean(nw, 120)
            ws = rolling_std(nw, 120)
            sq[w0:] = (nw - wm) / np.where(ws > 0, ws, 1.0)
        self.squeeze = sq
        self.don_hi30 = rollmax_prev(self.h, 30)
        self.don_lo30 = rollmin_prev(self.l, 30)
        self.ema1m20 = ema_series(self.c, 20)
        self.ret1 = np.concatenate(([0.0], self.c[1:] / self.c[:-1] - 1.0))
        # старшие TF (закрытые бары для каждого 1m)
        self.trend = {}
        rows = [[int(t), o, h, l, c, v] for t, o, h, l, c, v in
                zip(self.ts, self.o, self.h, self.l, self.c, self.v)]
        for tf_name, minutes in (('15m', 15), ('1h', 60)):
            agg = to_tf(rows, minutes)
            ctf = np.array([r[4] for r in agg])
            ttf = np.array([r[0] for r in agg], dtype=np.int64)
            e20 = ema_series(ctf, 20)
            e50 = ema_series(ctf, 50)
            # map: 1m ts -> индекс ЗАКРЫТОГО tf-бара
            mapf = np.full(n, -1, dtype=np.int64)
            cur, filled = -1, -1
            for i, t in enumerate(self.ts):
                b = int(t) // (minutes * 60000)
                if b != cur:
                    cur = b
                    filled = b - 1
                mapf[i] = filled
            # bucket id -> позиция в agg по timestamp открытия (устойчиво к дырам)
            want_ts = np.where(mapf >= 0, mapf, 0) * (minutes * 60000)
            pos = np.searchsorted(ttf, want_ts, side='right') - 1
            bad = (mapf < 0) | (pos < 0) | (pos >= ttf.size) | (ttf[np.clip(pos, 0, None)] != want_ts)
            pos[bad] = -1
            self.trend[tf_name] = dict(map=pos, c=ctf, ts=ttf, e20=e20, e50=e50)
        # VWAP от начала UTC-дня
        day = self.ts // DAY
        pv = self.c * self.v
        cs_pv = np.concatenate(([0.0], np.cumsum(pv)))
        cs_v = np.concatenate(([0.0], np.cumsum(self.v)))
        d0 = np.searchsorted(day, day, side='left')
        v_num = cs_pv[1:] - cs_pv[d0]
        v_den = cs_v[1:] - cs_v[d0]
        with np.errstate(invalid='ignore', divide='ignore'):
            self.vwap = np.where(v_den > 0, v_num / v_den, np.nan)

    def side_trend(self, i, tf):
        t = self.trend[tf]
        j = int(t['map'][i])
        if j < 0 or j >= t['c'].size or np.isnan(t['e20'][j]) or np.isnan(t['e50'][j]):
            return None
        if t['c'][j] > t['e20'][j] > t['e50'][j]:
            return 'long'
        if t['c'][j] < t['e20'][j] < t['e50'][j]:
            return 'short'
        return None


BTC = None


def masks(ld, cfg):
    """Семейства сигналов: dict idea -> (long_arr, short_arr) — Exclusive bool arrays."""
    n = ld.ts.size
    amin = float(cfg.get('atr_min_pct', 0.05))
    ok = np.isfinite(ld.atr_pct) & np.isfinite(ld.bb_up) & np.isfinite(ld.vol_base)
    base = ok & (ld.atr_pct >= amin)
    vr = np.where(np.isfinite(ld.vol_ratio), ld.vol_ratio, 0.0)
    vol_ok = vr >= cfg.get('vol_mult', 1.5)

    def side_arr(tf):
        t = ld.trend[tf]
        m = t['map']
        valid = m >= 0
        j = np.where(valid, m, 0)
        up = (t['c'][j] > t['e20'][j]) & (t['e20'][j] > t['e50'][j])
        dn = (t['c'][j] < t['e20'][j]) & (t['e20'][j] < t['e50'][j])
        return valid & np.nan_to_num(up, nan=False).astype(bool), \
               valid & np.nan_to_num(dn, nan=False).astype(bool)

    l15, s15 = side_arr('15m')
    l1h, s1h = side_arr('1h')
    sq = np.where(np.isfinite(ld.squeeze), ld.squeeze, 10.0)
    brk_up = ld.c > ld.bb_up
    brk_dn = ld.c < ld.bb_lo

    out = {}
    out['trend_bb15'] = ((l15 & brk_up & vol_ok & base), (s15 & brk_dn & vol_ok & base))
    out['trend_bb1h'] = ((l1h & brk_up & vol_ok & base), (s1h & brk_dn & vol_ok & base))
    # squeeze: сжатие (z<0) в последних 30 барах, затем BB-пробой по тренду 15m
    sq_bad = (sq < 0.0)
    csq = np.concatenate(([0], np.cumsum(sq_bad.astype(np.int64))))
    win = 30
    lo_idx = np.maximum(np.arange(n) - win, 0)
    recent_sq = (csq[1:] - csq[lo_idx]) > 0
    out['squeeze'] = ((recent_sq & brk_up & l15 & vol_ok & base),
                      (recent_sq & brk_dn & s15 & vol_ok & base))
    # mean reversion
    out['mr_rsi5'] = ((l1h & (ld.rsi5 < 25) & base), (s1h & (ld.rsi5 > 75) & base))
    out['mr_vwap'] = ((l1h & (ld.c < ld.vwap * (1 - 3 * EPS)) & base),
                      (s1h & (ld.c > ld.vwap * (1 + 3 * EPS)) & base))
    # Donchian-пробой по тренду 1h
    out['donchian'] = ((l1h & (ld.c > ld.don_hi30) & vol_ok & base),
                       (s1h & (ld.c < ld.don_lo30) & vol_ok & base))
    # импульс-продолжение (1m ход > 2*ATR%)
    tmove = 2.0 * np.nan_to_num(ld.atr_pct) / 100.0
    out['impulse'] = ((l15 & (ld.ret1 > tmove) & base),
                      (s15 & (ld.ret1 < -tmove) & base))
    # pullback к ema20(1m) в тренде 1h
    near = np.abs(ld.c / np.where(np.isfinite(ld.ema1m20), ld.ema1m20, 1.0) - 1.0) < 0.0015
    out['pullback'] = ((l1h & near & (ld.ret1 < 0) & base),
                       (s1h & near & (ld.ret1 > 0) & base))
    # BTC-led: альты идут за 1m-импульсом BTC
    if BTC is not None and BTC.symbol != ld.symbol:
        pos = np.searchsorted(BTC.ts, ld.ts)
        pv = np.minimum(pos, BTC.ts.size - 1)
        valid = (pos < BTC.ts.size) & (BTC.ts[pv] == ld.ts)
        bmap = np.where(valid, BTC.ret1[pv], 0.0)
        bthr = 2.0 * np.nan_to_num(ld.atr_pct) / 100.0 / 3.0
        out['btc_led'] = ((l15 & (bmap > bthr) & base),
                          (s15 & (bmap < -bthr) & base))
    return out


def sim(ld, cfg, i, side, t1_limit):
    """Универсальная симуляция сделки с режимом выхода."""
    o, h, l, c, ts = ld.o, ld.h, ld.l, ld.c, ld.ts
    atr1 = float(ld.atr[i])
    j = i + 1
    if j >= ts.size or (t1_limit and ts[j] > t1_limit):
        return None
    if side == 'long':
        entry = o[j] * (1 - EPS)
        if l[j] > entry:
            return None
    else:
        entry = o[j] * (1 + EPS)
        if h[j] < entry:
            return None
    sl_d = _sl_dist_generic(cfg, c[i], atr1)
    mode = cfg['exit']
    rr = cfg.get('rr', 2.5)
    if side == 'long':
        sl, tp = entry - sl_d, entry + rr * sl_d
    else:
        sl, tp = entry + sl_d, entry - rr * sl_d
    be_moved = False
    trail = None
    t_stop_ms = int(cfg.get('time_stop_min', 20)) * 60000
    exit_ = None
    trail_mult = cfg.get('trail_atr', 2.0)
    be_trigger = cfg.get('be_trigger', 1.0)
    j_end = min(j + 361, ts.size)
    for k in range(j, j_end):
        if mode == 'trail' and trail is not None:
            sl_now = trail
        elif mode == 'rr' and be_moved:
            sl_now = entry + cfg.get('be_offset', 0.0006) * entry if side == 'long' \
                else entry - cfg.get('be_offset', 0.0006) * entry
        else:
            sl_now = sl
        if side == 'long':
            hit_sl, hit_tp = l[k] <= sl_now, h[k] >= tp
        else:
            hit_sl, hit_tp = h[k] >= sl_now, l[k] <= tp
        if hit_sl:
            px = min(o[k], sl_now) if side == 'long' else max(o[k], sl_now)
            rsn = 'trail' if (mode == 'trail' and trail is not None) else \
                  ('be' if (mode == 'rr' and be_moved) else 'sl')
            exit_ = (px, rsn, k)
            break
        if hit_tp:
            px = max(o[k], tp) if side == 'long' else min(o[k], tp)
            exit_ = (px, 'tp', k)
            break
        profit_now = (h[k] - entry) if side == 'long' else (entry - l[k])
        if mode == 'trail':
            if trail is None:
                if profit_now >= be_trigger * sl_d:
                    trail = (h[k] - trail_mult * atr1) if side == 'long' \
                        else (l[k] + trail_mult * atr1)
            else:
                trail = max(trail, h[k] - trail_mult * atr1) if side == 'long' \
                    else min(trail, l[k] + trail_mult * atr1)
        elif mode == 'rr' and not be_moved:
            if profit_now >= be_trigger * sl_d:
                be_moved = True
            elif (ts[k] - ts[j]) >= t_stop_ms:
                exit_ = (c[k], 'time', k)
                break
        if mode == 'time' and (ts[k] - ts[j]) >= float(cfg.get('max_hold_min', 45)) * 60000:
            exit_ = (c[k], 'time', k)
            break
        if k == j_end - 1 and exit_ is None:
            exit_ = (c[k], 'cap', k)
    if exit_ is None:
        exit_ = (c[ts.size - 1], 'cap', ts.size - 1)
    exit_px, reason, exit_j = exit_
    dirn = 1.0 if side == 'long' else -1.0
    fee_out = MAKER if reason == 'tp' else TAKER
    fees_per_notional = MAKER + fee_out * exit_px / entry
    sl_pct = sl_d / entry
    pnl_per_qty = dirn * (exit_px - entry) - entry * fees_per_notional
    R = pnl_per_qty / (entry * sl_pct)
    return dict(symbol=ld.symbol, side=side, fees_R=round(fees_per_notional / sl_pct, 3),
                ts_entry=int(ts[j]), ts_exit=int(ts[exit_j]),
                entry=round(float(entry), 8), exit=round(float(exit_px), 8),
                exit_reason=reason, bars_held=exit_j - j + 1,
                R=round(float(R), 4), sl_pct=round(float(sl_pct), 6), minute=i)


def _sl_dist_generic(cfg, close, atr1):
    d = cfg.get('sl_atr_mult', 1.5) * atr1
    return min(max(d, close * cfg.get('sl_min_pct', 0.0015)),
               close * cfg.get('sl_max_pct', 0.0200))


def portfolio_accept(trades, cfg, t1):
    equity = 1000.0
    accepted = []
    busy_until = {}
    day, day_start, day_pnl, halted = None, 1000.0, 0.0, False
    for t in sorted(trades, key=lambda x: x['ts_entry']):
        if t['ts_entry'] > t1:
            break
        d = t['ts_entry'] // DAY
        if d != day:
            day = d
            day_start = equity
            day_pnl = 0.0
            halted = False
        sym = t['symbol']
        if halted or (sym in busy_until and busy_until[sym] > t['ts_entry']):
            continue
        if sum(1 for v in busy_until.values() if v > t['ts_entry']) >= cfg.get('max_positions', 4):
            continue
        risk = equity * cfg.get('risk_pct', 0.01)
        pnl = t['R'] * risk
        t2 = dict(t)
        t2['risk_usdt'] = round(risk, 4)
        t2['pnl'] = round(pnl, 4)
        t2['fees_usdt'] = round(t['fees_R'] * risk, 4)
        accepted.append(t2)
        busy_until[sym] = t['ts_exit']
        equity += pnl
        day_pnl += pnl
        if day_start > 0 and day_pnl <= -day_start * cfg.get('day_loss_pct', 0.04):
            halted = True
    return accepted, equity


def stats(trades):
    if not trades:
        return dict(n=0, sum_R=0.0, win_rate=0.0, expectancy_R=0.0, profit_factor=None,
                    net_pnl=0.0)
    R = np.array([t['R'] for t in trades], dtype=float)
    pnl = np.array([t.get('pnl', 0.0) for t in trades], dtype=float)
    w = R[R > 0]
    lo = R[R <= 0]
    return dict(n=int(R.size), sum_R=round(float(R.sum()), 2),
                win_rate=round(100.0 * float((R > 0).mean()), 1),
                expectancy_R=round(float(R.mean()), 4),
                profit_factor=round(float(w.sum() / abs(lo.sum())), 3)
                if lo.size and lo.sum() != 0 else None,
                net_pnl=round(float(pnl.sum()), 2))


def monthly_table(trades):
    mon = {}
    for t in trades:
        key = datetime.datetime.utcfromtimestamp(t['ts_entry'] / 1000).strftime('%Y-%m')
        mon.setdefault(key, []).append(t['R'])
    return {k: dict(n=len(v), sum_R=round(sum(v), 2)) for k, v in sorted(mon.items())}


def run_window(labs, mcache, idea, exit_cfg, lo, hi):
    """Кандидаты idea×exit в окне [lo, hi); последовательный гейт по символу."""
    trades = []
    cnt_cfg = dict(sl_min_pct=exit_cfg.get('sl_min_pct', 0.0015))
    for sym, ld in labs.items():
        key = (sym, idea)
        if key not in mcache:
            continue
        larr, sarr = mcache[key]
        mask = np.flatnonzero((larr | sarr) & (ld.ts >= lo) & (ld.ts < hi))
        next_free = -1
        sym_tr = []
        for i in mask:
            if i <= next_free:
                continue
            side = 'long' if larr[i] else 'short'
            t = sim(ld, exit_cfg, int(i), side, hi)
            if t is None:
                continue
            sym_tr.append(t)
            next_free = int(i) + t['bars_held']
        trades.extend(sym_tr)
    return trades


EXIT_CFGS = {
    'rr:1.5x2.5': dict(sl_atr_mult=1.5, rr=2.5, exit='rr'),
    'wide:3.5x3.0': dict(sl_atr_mult=3.5, rr=3.0, exit='rr', sl_max_pct=0.02),
    'trail:2.0': dict(sl_atr_mult=2.0, trail_atr=2.0, exit='trail'),
    'time:45m': dict(sl_atr_mult=1.5, exit='time', max_hold_min=45),
}
MASK_CFGS = {
    'vol1.5': dict(atr_min_pct=0.05, vol_mult=1.5),
    'vol1.8': dict(atr_min_pct=0.08, vol_mult=1.8),
    'loose': dict(atr_min_pct=0.03, vol_mult=1.0),
}


def _fmt(ms):
    return datetime.datetime.fromtimestamp(ms / 1000, datetime.timezone.utc).strftime('%Y-%m-%d')


def main():
    global BTC
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--symbols', default='')
    args = ap.parse_args()

    from config import SYMBOLS
    full = [s.replace('/', '').split(':')[0] for s in SYMBOLS]
    if args.symbols:
        chosen = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    elif args.quick:
        chosen = [s for s in full if s.startswith(('BTC', 'SOL'))]
    else:
        chosen = full
    chosen = sorted(set(chosen), key=full.index if full else (lambda s: 0))

    labs = {}
    for i, sym in enumerate(chosen):
        bars = load_1m(sym, days=190)
        if not bars:
            print('skip %s (no data)' % sym, flush=True)
            continue
        ld = LabData(sym, bars)
        labs[sym] = ld
        if sym.startswith('BTC'):
            BTC = ld
        print('[data %d/%d] %s: %d bars (%d days)' % (
            i + 1, len(chosen), sym, len(bars),
            round((bars[-1][0] - bars[0][0]) / DAY)), flush=True)
    if not labs:
        print('no data loaded', flush=True)
        return

    t_start = int(min(ld.ts[0] for ld in labs.values()))
    t_end = int(max(ld.ts[-1] for ld in labs.values()))
    is_end = t_start + 120 * DAY
    oos_start = t_end - 60 * DAY
    print('IS: %s .. %s | OOS: %s .. %s' % (
        _fmt(t_start), _fmt(is_end), _fmt(oos_start), _fmt(t_end)), flush=True)

    results = []
    step = 0
    for mc_name, mcfg in MASK_CFGS.items():
        mcache = {}
        for sym, ld in labs.items():
            M = masks(ld, mcfg)
            for idea, (la, sa) in M.items():
                mcache[(sym, idea)] = (la, sa)
        ideas = sorted({k[1] for k in mcache})
        for idea in ideas:
            for ex_name, ex in EXIT_CFGS.items():
                step += 1
                tr_is = run_window(labs, mcache, idea, ex, t_start, is_end)
                acc_is, eq_is = portfolio_accept(tr_is, ex, is_end)
                st_is = stats(acc_is)
                row = dict(mask=mc_name, idea=idea, exit=ex_name, IS=st_is)
                ok = st_is['n'] >= 30 and st_is['sum_R'] > 0
                if ok:
                    tr_oos = run_window(labs, mcache, idea, ex, oos_start, t_end)
                    acc_oos, eq_oos = portfolio_accept(tr_oos, ex, t_end)
                    row['OOS'] = stats(acc_oos)
                    acc_full, _ = portfolio_accept(
                        run_window(labs, mcache, idea, ex, t_start, t_end), ex, t_end)
                    row['monthly'] = monthly_table(acc_full)
                results.append(row)
                print('[%03d] %-6s %-10s %-12s IS n=%d sumR=%.1f%s' % (
                    step, mc_name, idea, ex_name, st_is['n'], st_is['sum_R'],
                    (' | OOS n=%d sumR=%.1f' % (row['OOS']['n'], row['OOS']['sum_R']))
                    if 'OOS' in row else ' -> cut'), flush=True)

    # ---------- per-symbol скан: ищем монеты с локальным эджем ----------
    per_sym = []
    for sym, ld in labs.items():
        labs1 = {sym: ld}
        for mc_name, mcfg in MASK_CFGS.items():
            M1 = masks(ld, mcfg)
            for idea, (la, sa) in M1.items():
                mcache1 = {(sym, idea): (la, sa)}
                for ex_name, ex in EXIT_CFGS.items():
                    acc_is, _ = portfolio_accept(
                        run_window(labs1, mcache1, idea, ex, t_start, is_end), ex, is_end)
                    st = stats(acc_is)
                    if st['n'] < 25:
                        continue
                    acc_oos, _ = portfolio_accept(
                        run_window(labs1, mcache1, idea, ex, oos_start, t_end), ex, t_end)
                    row = dict(symbol=sym, mask=mc_name, idea=idea, exit=ex_name,
                               IS=st, OOS=stats(acc_oos))
                    per_sym.append(row)
        print('[per-sym done] %s (variants kept: %d)' % (sym, len(per_sym)), flush=True)

    is_pos = [r for r in results if r['IS']['sum_R'] > 0]
    is_pos.sort(key=lambda r: r['IS']['sum_R'], reverse=True)
    oos_ok = [r for r in results if r.get('OOS') and r['OOS']['sum_R'] > 0]
    oos_ok.sort(key=lambda r: r['OOS']['sum_R'], reverse=True)

    out = dict(protocol='IS 120d / OOS 60d', t_start=t_start, t_end=t_end,
               symbols=sorted(labs), variants=len(results),
               top_is=is_pos[:15], top_oos=oos_ok[:15], all_results=results,
               per_symbol=per_sym)
    os.makedirs('research/results', exist_ok=True)
    with open('research/results/lab.json', 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    # ---------- отчёт ----------
    funding_sec = ''
    fp = 'research/results/funding.json'
    if os.path.exists(fp):
        try:
            fu = json.load(open(fp))
            lines = ['| symbol | mean funding | std |', '|---|---|---|']
            for s, v in sorted(fu.items()):
                if 'mean' in v and v.get('mean') is not None:
                    lines.append('| %s | %.4f%% | %.4f%% |' % (s, v['mean'] * 100, (v.get('std') or 0) * 100))
            funding_sec = '\n## Funding-контекст (Bybit, последние 200 периодов)\n\n' + '\n'.join(lines) + '\n'
        except Exception as e:
            funding_sec = '\n(funding.json не прочитан: %s)\n' % e

    rep = ['# Перебор стратегий (lab.py) — IS 120д/OOS 60д\n']
    rep.append('Символов: %d. Вариантов idea×exit×mask: %d.\n' % (len(labs), len(results)))
    rep.append('\n## Топ-10 по IS (отбор только по IS)\n')
    rep.append('| mask | idea | exit | n | sum_R | WR% | PF | OOS n | OOS sum_R |')
    rep.append('|---|---|---|---|---|---|---|---|---|')
    for r in is_pos[:10]:
        o = r.get('OOS', {})
        rep.append('| %s | %s | %s | %d | %.1f | %.1f | %s | %s | %s |' % (
            r['mask'], r['idea'], r['exit'], r['IS']['n'], r['IS']['sum_R'],
            r['IS']['win_rate'], r['IS']['profit_factor'],
            o.get('n', '-'), o.get('sum_R', '-')))
    rep.append('\n## Проверка OOS (положительные на чистых данных)\n')
    if oos_ok:
        for r in oos_ok[:15]:
            rep.append('- %s/%s/%s: OOS n=%d sum_R=%.1f WR=%.1f%%' % (
                r['mask'], r['idea'], r['exit'], r['OOS']['n'],
                r['OOS']['sum_R'], r['OOS']['win_rate']))
            if 'monthly' in r:
                rep.append('  - месяцы: ' + ', '.join(
                    '%s %+.1fR(n=%d)' % (k, v['sum_R'], v['n'])
                    for k, v in r['monthly'].items()))
    else:
        rep.append('**Ни один вариант idea×exit не прошёл OOS.** Устойчивого эджа не найдено.')
    ver = '\n## Вердикт\n'
    if oos_ok:
        best = oos_ok[0]
        months = best.get('monthly', {})
        pos_m = sum(1 for v in months.values() if v['sum_R'] > 0)
        ver += ('\nЛучший OOS-вариант: **%s/%s/%s** (sum_R %.1f, %d/%d месяцев в плюсе).\n\n'
                'Дальше: per-symbol разбор и мини-live наблюдение. Критерий для реала — '
                'OOS-плюс и большинство месяцев в плюсе.\n' % (
                    best['mask'], best['idea'], best['exit'], best['OOS']['sum_R'],
                    pos_m, len(months)))
    else:
        ver += ('\nЭджа нет: `не выходить на реальные деньги`. Текущая стратегия всегда '
                'проверяется тем же протоколом IS/OOS перед любым изменением.\n')
    rep.append(ver)
    rep.append(funding_sec)
    ps_pos = [r for r in per_sym if r['IS']['sum_R'] > 0 and r['OOS']['sum_R'] > 0]
    ps_pos.sort(key=lambda r: r['OOS']['sum_R'], reverse=True)
    rep.append('\n## Per-symbol: монеты с положительным И IS, И OOS\n')
    if ps_pos:
        rep.append('| symbol | mask | idea | exit | IS n | IS sum_R | OOS n | OOS sum_R | WR OOS% |')
        rep.append('|---|---|---|---|---|---|---|---|---|')
        for r in ps_pos[:20]:
            rep.append('| %s | %s | %s | %s | %d | %.1f | %d | %.1f | %.1f |' % (
                r['symbol'], r['mask'], r['idea'], r['exit'],
                r['IS']['n'], r['IS']['sum_R'], r['OOS']['n'], r['OOS']['sum_R'],
                r['OOS']['win_rate']))
    else:
        rep.append('Ни одна монета не прошла оба этапа (IS+ и OOS+).')
    rep.append('\nВсего per-symbol вариантов с n>=25: %d, из них IS+: %d, IS+ и OOS+: %d.\n' % (
        len(per_sym), sum(1 for r in per_sym if r['IS']['sum_R'] > 0), len(ps_pos)))
    with open('research/results/lab.md', 'w') as f:
        f.write('\n'.join(rep) + '\n')
    print('done. is_pos=%d oos_ok=%d per_sym_all=%d per_sym_pos=%d' % (
        len(is_pos), len(oos_ok), len(per_sym), len(ps_pos)), flush=True)


if __name__ == '__main__':
    main()
