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
    """Предрасчёт индикаторов по символу (общий для всех комбинаций параметров)."""

    def __init__(self, symbol, bars):
        self.symbol = symbol
        self.bars = bars
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

    def check_signal(self, i):
        import strategy
        a1 = max(0, i + 2 - CANDLE_LIMIT)
        df1 = self.bars[a1:i + 2]
        a5 = max(0, int(self.map5[i]) + 1 - CANDLE_LIMIT)
        df5 = to_5m(self.bars[a5:i + 1]) + [list(self.bars[i])]
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


def _new_position(sd, cfg, p, side, atr1, equity):
    """Заполнение заявки (entry bar = p). None — если вход не состоялся."""
    o, h, l = sd.o, sd.h, sd.l
    eps = cfg['entry_eps']
    if side == 'long':
        entry = o[p] * (1.0 - eps)
        if l[p] > entry:
            return None
    else:
        entry = o[p] * (1.0 + eps)
        if h[p] < entry:
            return None
    risk_usdt = equity * cfg['risk_pct']
    sl_d = _sl_dist(cfg, entry, atr1)
    qty = risk_usdt / sl_d
    notional = qty * entry
    cap = equity * 10.0 * 0.95
    if notional > cap:
        qty = cap / entry
    if qty <= 0:
        return None
    if side == 'long':
        sl, tp = entry - sl_d, entry + cfg['rr'] * sl_d
        be_trigger = entry + cfg['be_trigger'] * sl_d
        be_sl = entry + cfg['be_offset_pct'] * entry
    else:
        sl, tp = entry + sl_d, entry - cfg['rr'] * sl_d
        be_trigger = entry - cfg['be_trigger'] * sl_d
        be_sl = entry - cfg['be_offset_pct'] * entry
    tr = dict(symbol=sd.symbol, side=side, ts_entry=int(sd.ts[p]),
              entry=entry, qty=qty, sl=sl, tp=tp, sl_d=sl_d,
              be_trigger=be_trigger, be_sl=be_sl, be_moved=False,
              risk_usdt=risk_usdt, atr1=float(atr1))
    return tr


def _bar_exit_check(tr, sd, p, cfg, first_bar=False):
    """p — бар, только что закрытый. Возвращает (exit_px|None, reason|None)."""
    side = tr['side']
    o, h, l, c = sd.o[p], sd.h[p], sd.l[p], sd.c[p]
    sl_now = tr['be_sl'] if tr['be_moved'] else tr['sl']
    if side == 'long':
        if l <= sl_now:
            return min(o, sl_now), ('be' if tr['be_moved'] else 'sl')
        if h >= tr['tp']:
            return max(o, tr['tp']), 'tp'
        if not tr['be_moved'] and h >= tr['be_trigger']:
            tr['be_moved'] = True
            return None, None
    else:
        if h >= sl_now:
            return max(o, sl_now), ('be' if tr['be_moved'] else 'sl')
        if l <= tr['tp']:
            return min(o, tr['tp']), 'tp'
        if not tr['be_moved'] and l <= tr['be_trigger']:
            tr['be_moved'] = True
            return None, None
    if first_bar:
        return None, None
    if not tr['be_moved'] and (sd.ts[p] - tr['ts_entry']) >= int(cfg['time_stop_min']) * 60000:
        return c, 'time'
    if (sd.ts[p] - tr['ts_entry']) > 360 * 60000:
        return c, 'cap'
    return None, None


def _finalize(tr, sd, p, exit_px, reason, cfg):
    dirn = 1.0 if tr['side'] == 'long' else -1.0
    qty = tr['qty']
    fee_out = cfg['maker_fee'] if reason == 'tp' else cfg['taker_fee']
    fees = cfg['maker_fee'] * tr['entry'] * qty + fee_out * exit_px * qty
    pnl = dirn * (exit_px - tr['entry']) * qty - fees
    out = dict(symbol=tr['symbol'], side=tr['side'],
               ts_entry=tr['ts_entry'], ts_exit=int(sd.ts[p]),
               entry=round(tr['entry'], 8), exit=round(exit_px, 8),
               sl=round(tr['sl'], 8), tp=round(tr['tp'], 8),
               qty=round(qty, 8), exit_reason=reason,
               bars_held=int((sd.ts[p] - tr['ts_entry']) // 60000),
               risk_usdt=round(tr['risk_usdt'], 4), fees=round(fees, 4),
               pnl=round(pnl, 4), R=round(pnl / tr['risk_usdt'], 4),
               feats=tr.get('feats'))
    return out


def run(cfg=None, symbols=None, refresh=False, t0=0, t1=None, datas=None):
    cfg = cfg or default_cfg()
    if symbols is None:
        import config as C
        symbols = C.SYMBOLS
    if datas is None:
        datas = [SymbolData(s, load_1m(s, refresh=refresh)) for s in symbols]
    else:
        datas = [d for d in datas if d.symbol in set(symbols)] if symbols else datas
    for sd in datas:
        if sd.ts.size < 200:
            raise RuntimeError('%s: мало данных (%d баров)' % (sd.symbol, sd.ts.size))
    start_equity = 1000.0
    equity = start_equity
    trades = []
    open_pos = {}
    pending = {}
    day = None
    day_start = equity
    day_pnl = 0.0
    halted = False
    t_start = max(int(sd.ts[25]) for sd in datas)
    t_end = min(int(sd.ts[-2]) for sd in datas)
    if t1:
        t_end = min(t_end, t1)
    times = sorted({int(t) for sd in datas for t in sd.ts if t_start <= t <= t_end})
    idx = {sd.symbol: int(np.searchsorted(sd.ts, max(t_start, t0 or 0)))
           for sd in datas}
    for T in times:
        if T < t0:
            continue
        d = T // 86400000
        if d != day:
            day = d
            day_start = equity
            day_pnl = 0.0
            halted = False
        minute_signals = []
        for sd in datas:
            sym = sd.symbol
            p = idx[sym]
            if p >= sd.ts.size or int(sd.ts[p]) != T:
                continue
            pd_entry = pending.pop(sym, None)
            if pd_entry is not None and not halted and sym not in open_pos \
                    and len(open_pos) < cfg['max_positions']:
                _, p_sig, side, atr1, eq_at_signal = pd_entry
                tr = _new_position(sd, cfg, p, side, atr1, equity)
                if tr is not None:
                    tr['feats'] = sd.feats(p_sig, side, atr1)
                    open_pos[sym] = tr
                    ex, reason = _bar_exit_check(tr, sd, p, cfg, first_bar=True)
                    if ex is not None:
                        fin = _finalize(tr, sd, p, ex, reason, cfg)
                        equity += fin['pnl']
                        day_pnl += fin['pnl']
                        trades.append(fin)
                        del open_pos[sym]
            elif sym in open_pos:
                tr = open_pos[sym]
                ex, reason = _bar_exit_check(tr, sd, p, cfg)
                if ex is not None:
                    fin = _finalize(tr, sd, p, ex, reason, cfg)
                    equity += fin['pnl']
                    day_pnl += fin['pnl']
                    trades.append(fin)
                    del open_pos[sym]
            idx[sym] = p + 1
        if day_start > 0 and day_pnl <= -day_start * cfg['day_loss_pct'] and not halted:
            halted = True
            for sym, tr in list(open_pos.items()):
                sd2 = next(x for x in datas if x.symbol == sym)
                p2 = idx[sym] - 1
                if 0 <= p2 < sd2.ts.size:
                    fin = _finalize(tr, sd2, p2, float(sd2.c[p2]), 'day', cfg)
                    equity += fin['pnl']
                    day_pnl += fin['pnl']
                    trades.append(fin)
                del open_pos[sym]
            pending = {}
        if halted or len(open_pos) >= cfg['max_positions']:
            continue
        for sd in datas:
            sym = sd.symbol
            p = idx[sym] - 1
            if sym in open_pos or sym in pending:
                continue
            if p < 25 or p >= sd.ts.size - 1 or int(sd.ts[p]) != T:
                continue
            ap = sd.atr_pct[p]
            if not np.isfinite(ap) or ap < cfg['atr_min_pct']:
                continue
            vb = sd.vol_base[p]
            if not np.isfinite(vb) or vb <= 0:
                continue
            if sd.v[p] < cfg['vol_mult'] * vb:
                continue
            side5 = sd.trend_side(p)
            rsi = sd.rsi[p]
            if side5 is None or not np.isfinite(rsi):
                continue
            if side5 == 'long':
                if not (cfg['rsi_long_lo'] <= rsi <= cfg['rsi_long_hi']):
                    continue
                upper = sd.bb_mid[p] + cfg['bb_k'] * sd.bb_sd[p]
                trig = sd.c[p] > upper or sd.c[p] > sd.brk_hi[p]
            else:
                if not (cfg['rsi_short_lo'] <= rsi <= cfg['rsi_short_hi']):
                    continue
                lower = sd.bb_mid[p] - cfg['bb_k'] * sd.bb_sd[p]
                trig = sd.c[p] < lower or sd.c[p] < sd.brk_lo[p]
            if not trig:
                continue
            sig = sd.check_signal(p)
            if sig is None or sig[0] != side5:
                continue
            _, atr1, _ = sig
            pending[sym] = (T, p, side5, atr1, equity)
    return _report(trades, start_equity, equity), trades


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
