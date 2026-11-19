"""research/backtest.py — событийный бэктест, использующий ровно ту же
check_signal/индикаторы/риск-правила, что и live-бот (strategy.py, config.py).

Модель исполнения (консервативная):
- вход: postOnly-лимит = open следующего бара с буфером ENTRY_EPS (3 бп по умолч.),
  вход засчитывается, только если бар реально прошел эту цену;
- если в баре затронуты и SL и TP — считаем, что сначала сработал SL;
- выход по SL / тайм-стопу / кэпу / дневному стопу — taker-fee, по TP — maker-fee;
- безубыток (BE) при BE_TRIGGER.R, тайм-стоп TIME_STOP_MIN минут без BE,
  дневной стоп-аут DAY_LOSS_PCT — как в trader.py/risk_manager.py.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.data import load_1m, to_5m
from research.indicators_vec import (ema_series, rollmax_prev, rollmin_prev,
                                     rolling_mean, rolling_std, wilder_atr,
                                     wilder_rsi)

CANDLE_LIMIT = 100


def default_cfg():
    import config as C
    return dict(
        atr_min_pct=C.ATR_MIN_PCT, vol_mult=C.VOL_MULT, rr=C.RR, bb_k=C.BB_K,
        rsi_long_lo=C.RSI_LONG[0], rsi_long_hi=C.RSI_LONG[1],
        rsi_short_lo=C.RSI_SHORT[0], rsi_short_hi=C.RSI_SHORT[1],
        sl_atr_mult=C.SL_ATR_MULT, sl_min_pct=C.SL_MIN_PCT,
        sl_max_pct=C.SL_MAX_PCT, be_trigger=C.BE_TRIGGER,
        be_offset_pct=C.BE_OFFSET_PCT, time_stop_min=C.TIME_STOP_MIN,
        max_positions=C.MAX_POSITIONS, risk_pct=C.RISK_PER_TRADE_PCT,
        day_loss_pct=C.DAY_LOSS_PCT,
        maker_fee=0.0002, taker_fee=0.00055, entry_eps=0.0003,
    )


class SymbolData:
    """Предрасчёт индикаторов по символу (общий для всех комбинаций параметров).
    Хранит только numpy-массивы (экономия памяти при длинной истории)."""

    def __init__(self, symbol, bars):
        self.symbol = symbol
        self.ts = np.array([b[0] for b in bars], dtype=np.int64)
        self.o = np.array([b[1] for b in bars], dtype=float)
        self.h = np.array([b[2] for b in bars], dtype=float)
        self.l = np.array([b[3] for b in bars], dtype=float)
        self.c = np.array([b[4] for b in bars], dtype=float)
        self.v = np.array([b[5] for b in bars], dtype=float)
        b5 = to_5m(bars)
        self.c5 = np.array([b[4] for b in b5], dtype=float)
        self.ts5 = np.array([b[0] for b in b5], dtype=np.int64)
        self.ema5 = ema_series(self.c5, 21)
        self.atr_pct = wilder_atr(self.h, self.l, self.c) / self.c * 100.0
        self.rsi = wilder_rsi(self.c)
        self.vol_base = rolling_mean(self.v, 20, shift=1)
        self.bb_mid = rolling_mean(self.c, 20)
        self.bb_sd = rolling_std(self.c, 20)
        self.brk_hi = rollmax_prev(self.h, 5)
        self.brk_lo = rollmin_prev(self.l, 5)
        self.map5 = np.full(self.ts.size, -1, dtype=np.int64)
        b0 = int(self.ts[0]) // 300000
        cur, filled = -1, -1
        for i, t in enumerate(self.ts):
            b = int(t) // 300000
            if b != cur:
                cur = b
                filled = b - 1 - b0
            self.map5[i] = filled

    def trend_side(self, i):
        j = int(self.map5[i])
        if j < 20 or j >= self.c5.size or np.isnan(self.ema5[j]):
            return None
        if self.c5[j] > self.ema5[j]:
            return 'long'
        if self.c5[j] < self.ema5[j]:
            return 'short'
        return None

    def _rows(self, a, b):
        ts, o, h, l, c, v = self.ts, self.o, self.h, self.l, self.c, self.v
        return [[int(ts[j]), o[j], h[j], l[j], c[j], v[j]] for j in range(a, b)]

    def check_signal(self, i):
        import strategy
        a1 = max(0, i + 2 - CANDLE_LIMIT)
        df1 = self._rows(a1, i + 2)
        a5 = max(0, int(self.map5[i]) + 1 - CANDLE_LIMIT)
        df5 = to_5m(self._rows(a5, i + 1)) + [self._rows(i, i + 1)[0]]
        return strategy.check_signal(df5, df1)

    def feats(self, i, side, atr1):
        f = dict(
            atr_pct=round(float(self.atr_pct[i]), 4),
            rsi=round(float(self.rsi[i]), 2),
            vol_ratio=round(float(self.v[i] / self.vol_base[i]), 3),
            side=side,
            hour=int((int(self.ts[i]) // 3600000) % 24),
            trend5=round(float(self.c5[int(self.map5[i])] / self.ema5[int(self.map5[i])] - 1.0) * 100, 4),
        )
        if side == 'long':
            up = self.bb_mid[i] + 2.0 * self.bb_sd[i]
            f['bb_over'] = round(float((self.c[i] - up) / atr1), 3)
            f['brk_over'] = round(float((self.c[i] - self.brk_hi[i]) / atr1), 3)
        else:
            lo = self.bb_mid[i] - 2.0 * self.bb_sd[i]
            f['bb_over'] = round(float((lo - self.c[i]) / atr1), 3)
            f['brk_over'] = round(float((self.brk_lo[i] - self.c[i]) / atr1), 3)
        return f


def _sl_dist(cfg, close, atr1):
    d = cfg['sl_atr_mult'] * atr1
    return min(max(d, close * cfg['sl_min_pct']), close * cfg['sl_max_pct'])


def candidates(sd, cfg, t0=0, t1=None):
    """Векторный предфильтр + точный check_signal + симуляция сделки.
    Возвращает список сделок по одному символу (R на единичном риске)."""
    ts = sd.ts
    n = ts.size
    lo = 25
    hi = n - 2
    if t0:
        lo = max(lo, int(np.searchsorted(ts, t0)))
    if t1:
        hi = min(hi, int(np.searchsorted(ts, t1, 'right')) - 1)
    if hi <= lo:
        return []
    atr_ok = np.isfinite(sd.atr_pct[lo:hi + 1]) & (sd.atr_pct[lo:hi + 1] >= cfg['atr_min_pct'])
    vb = sd.vol_base[lo:hi + 1]
    vol_ok = np.isfinite(vb) & (vb > 0) & (sd.v[lo:hi + 1] >= cfg['vol_mult'] * vb)
    idx5 = sd.map5[lo:hi + 1]
    j5 = np.where(idx5 >= 20, idx5, 0)
    ema_ok = np.isfinite(sd.ema5[j5]) & (idx5 >= 20)
    c5 = sd.c5[j5]
    e5 = sd.ema5[j5]
    side_long = c5 > e5
    side_short = c5 < e5
    rsi = sd.rsi[lo:hi + 1]
    rsi_l = np.isfinite(rsi) & (rsi >= cfg['rsi_long_lo']) & (rsi <= cfg['rsi_long_hi'])
    rsi_s = np.isfinite(rsi) & (rsi >= cfg['rsi_short_lo']) & (rsi <= cfg['rsi_short_hi'])
    cc = sd.c[lo:hi + 1]
    upper = sd.bb_mid[lo:hi + 1] + cfg['bb_k'] * sd.bb_sd[lo:hi + 1]
    lower = sd.bb_mid[lo:hi + 1] - cfg['bb_k'] * sd.bb_sd[lo:hi + 1]
    trig_l = (cc > upper) | (cc > sd.brk_hi[lo:hi + 1])
    trig_s = (cc < lower) | (cc < sd.brk_lo[lo:hi + 1])
    mask = ((atr_ok & vol_ok & ema_ok & np.isfinite(upper) & np.isfinite(sd.brk_hi[lo:hi + 1]))
            & ((side_long & rsi_l & trig_l) | (side_short & rsi_s & trig_s)))
    cand = np.where(mask)[0] + lo
    trades = []
    strict = os.getenv('RESEARCH_STRICT', '0') == '1'
    for i in cand:
        i = int(i)
        if trades and ts[i + 1] < trades[-1]['ts_exit']:
            continue
        side = 'long' if sd.trend_side(i) == 'long' else 'short'
        if strict:
            sig = sd.check_signal(i)
            if sig is None or (sig[0] == 'long') != (side == 'long'):
                continue
            _, atr1, _ = sig
        else:
            atr1 = float(sd.atr_pct[i]) / 100.0 * float(sd.c[i])
        tr = _sim_from_signal(sd, cfg, i, side, atr1)
        if tr is not None:
            trades.append(tr)
    return trades


def _sim_from_signal(sd, cfg, i, side, atr1):
    """Сигнал на закрытии бара i; вход на баре i+1; walk до выхода.
    SL считается от close сигнального бара (как в трейдере), привязан к entry."""
    eps = cfg['entry_eps']
    o, h, l, c, ts = sd.o, sd.h, sd.l, sd.c, sd.ts
    j = i + 1
    if side == 'long':
        entry = o[j] * (1.0 - eps)
        if l[j] > entry:
            return None
    else:
        entry = o[j] * (1.0 + eps)
        if h[j] < entry:
            return None
    sl_d = _sl_dist(cfg, c[i], atr1)
    rr = cfg['rr']
    if side == 'long':
        sl, tp = entry - sl_d, entry + rr * sl_d
        trig = entry + cfg['be_trigger'] * sl_d
        be_sl = entry + cfg['be_offset_pct'] * entry
    else:
        sl, tp = entry + sl_d, entry - rr * sl_d
        trig = entry - cfg['be_trigger'] * sl_d
        be_sl = entry - cfg['be_offset_pct'] * entry
    be_moved = False
    exit_px = exit_j = reason = None
    t_stop_ms = int(cfg['time_stop_min']) * 60000
    for k in range(j, min(j + 361, ts.size)):
        sl_now = be_sl if be_moved else sl
        if side == 'long':
            hit_sl = l[k] <= sl_now
            hit_tp = h[k] >= tp
        else:
            hit_sl = h[k] >= sl_now
            hit_tp = l[k] <= tp
        if hit_sl:
            exit_px = min(o[k], sl_now) if side == 'long' else max(o[k], sl_now)
            reason, exit_j = ('be' if be_moved else 'sl'), k
            break
        if hit_tp:
            exit_px = max(o[k], tp) if side == 'long' else min(o[k], tp)
            reason, exit_j = 'tp', k
            break
        if not be_moved:
            if (side == 'long' and h[k] >= trig) or (side == 'short' and l[k] <= trig):
                be_moved = True
            elif (ts[k] - ts[j]) >= t_stop_ms:
                exit_px, reason, exit_j = c[k], 'time', k
                break
        if k == j + 360:
            exit_px, reason, exit_j = c[k], 'cap', k
            break
    if exit_j is None:
        return None
    dirn = 1.0 if side == 'long' else -1.0
    fee_out = cfg['maker_fee'] if reason == 'tp' else cfg['taker_fee']
    fees_per_notional = cfg['maker_fee'] + fee_out * exit_px / entry
    pnl_per_qty = dirn * (exit_px - entry) - entry * fees_per_notional
    sl_pct_of_price = sl_d / entry
    R = pnl_per_qty / (entry * sl_pct_of_price)
    feats = sd.feats(i, side, atr1)
    return dict(symbol=sd.symbol, side=side, fees_R=fees_per_notional / sl_pct_of_price,
                ts_entry=int(ts[j]), ts_exit=int(ts[exit_j]),
                entry=round(float(entry), 8), exit=round(float(exit_px), 8),
                sl=round(float(sl), 8), tp=round(float(tp), 8),
                exit_reason=reason, bars_held=exit_j - j + 1,
                R=round(float(R), 4), feats=feats)


def run(cfg=None, symbols=None, refresh=False, t0=0, t1=None, datas=None):
    cfg = cfg or default_cfg()
    if symbols is None:
        import config as C
        symbols = C.SYMBOLS
    smap = set(symbols)
    if datas is None:
        datas = [SymbolData(s, load_1m(s, refresh=refresh)) for s in symbols]
    else:
        datas = [d for d in datas if d.symbol in smap]
    start_equity = 1000.0
    equity = start_equity
    accepted = []
    busy_until = {}
    day = None
    day_start = equity
    day_pnl = 0.0
    halted = False
    t_end = min(int(d.ts[-2]) for d in datas)
    t1 = min(t1, t_end) if t1 else t_end

    all_tr = []
    for sd in datas:
        all_tr.extend(candidates(sd, cfg, t0=t0, t1=t1))
    all_tr.sort(key=lambda t: t['ts_entry'])

    for t in all_tr:
        if t['ts_entry'] >= t1:
            break
        d = t['ts_entry'] // 86400000
        if d != day:
            day = d
            day_start = equity
            day_pnl = 0.0
            halted = False
        sym = t['symbol']
        if halted or sym in busy_until and busy_until[sym] > t['ts_entry']:
            continue
        if sum(1 for v in busy_until.values() if v > t['ts_entry']) >= cfg['max_positions']:
            continue
        accepted.append(t)
        busy_until[sym] = t['ts_exit']
        risk_usdt = equity * cfg['risk_pct']
        pnl = t['R'] * risk_usdt
        fees = t['fees_R'] * risk_usdt
        t2 = dict(t)
        t2['risk_usdt'] = round(risk_usdt, 4)
        t2['pnl'] = round(pnl, 4)
        t2['fees'] = round(fees, 4)
        accepted[-1] = t2
        equity += pnl
        day_pnl += pnl
        if day_start > 0 and day_pnl <= -day_start * cfg['day_loss_pct']:
            halted = True
    rep = _report(accepted, start_equity, equity)
    return rep, accepted


def _report(trades, start_equity, end_equity):
    pnl = np.array([t['pnl'] for t in trades], dtype=float) if trades else np.array([])
    rrr = np.array([t['R'] for t in trades], dtype=float) if trades else np.array([])
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    eq = start_equity + np.cumsum(pnl) if trades else np.array([start_equity])
    peak = np.maximum.accumulate(eq)
    dd = ((peak - eq) / peak * 100.0) if trades else np.array([])
    pf = round(float(wins.sum() / abs(losses.sum())), 3) if losses.size and losses.sum() != 0 else None
    return dict(
        start_equity=start_equity, end_equity=round(end_equity, 2),
        net_pnl=round(end_equity - start_equity, 2),
        net_pnl_pct=round((end_equity - start_equity) / start_equity * 100.0, 2),
        n_trades=int(pnl.size),
        win_rate=round(100.0 * wins.size / pnl.size, 1) if pnl.size else 0.0,
        expectancy_R=round(float(rrr.mean()), 4) if pnl.size else 0.0,
        sum_R=round(float(rrr.sum()), 2) if pnl.size else 0.0,
        profit_factor=pf,
        max_dd_pct=round(float(dd.max()), 2) if trades else 0.0,
        sharpe_trade=round(float(rrr.mean() / rrr.std() * np.sqrt(rrr.size)), 3)
        if pnl.size > 2 and rrr.std() > 0 else 0.0,
        total_fees=round(float(sum(t['fees'] for t in trades)), 2) if trades else 0.0,
        by_reason={r: int(sum(1 for t in trades if t['exit_reason'] == r))
                   for r in ('tp', 'sl', 'be', 'time', 'day', 'cap')},
        by_symbol={},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=7)
    ap.add_argument('--refresh', action='store_true')
    ap.add_argument('--atr', type=float, default=None)
    ap.add_argument('--rr', type=float, default=None)
    ap.add_argument('--vol', type=float, default=None)
    ap.add_argument('--symbols', type=str, default=None)
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()
    cfg = default_cfg()
    if args.atr is not None:
        cfg['atr_min_pct'] = args.atr
    if args.rr is not None:
        cfg['rr'] = args.rr
    if args.vol is not None:
        cfg['vol_mult'] = args.vol
    symbols = [s.strip() for s in args.symbols.split(',') if s.strip()] if args.symbols else None
    rep, trades = run(cfg, symbols=symbols)
    rep['by_symbol'] = {}
    for t in trades:
        s = rep['by_symbol'].setdefault(t['symbol'], dict(n=0, pnl=0.0, wins=0))
        s['n'] += 1
        s['pnl'] = round(s['pnl'] + t['pnl'], 2)
        s['wins'] += 1 if t['pnl'] > 0 else 0
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(dict(report=rep, trades=trades), f, ensure_ascii=False)


if __name__ == '__main__':
    main()
