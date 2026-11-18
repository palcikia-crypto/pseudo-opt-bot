"""strategy.py — импульсные сигналы 1m (пробой BB/канала) с тренд-фильтром 5m."""
import numpy as np

from config import (ATR_MIN_PCT, ATR_N, BB_K, BB_N, BREAK_N, OB_DEPTH, OB_IMB,
                    RSI_LONG, RSI_N, RSI_SHORT, TF_TREND, VOL_MULT, VOL_SMA_N)
from indicators import (atr_last, breakout_high, breakout_low, bollinger_last,
                        ema_last, ob_imbalance, rsi_last, vol_sma_prev)


def _cols(ohlcv, closed_only=True):
    rows = list(ohlcv)
    if closed_only and len(rows) > 1:
        rows = rows[:-1]
    ts = np.array([r[0] for r in rows], dtype=np.int64)
    o = np.array([float(r[1]) for r in rows])
    h = np.array([float(r[2]) for r in rows])
    l = np.array([float(r[3]) for r in rows])
    c = np.array([float(r[4]) for r in rows])
    v = np.array([float(r[5]) for r in rows])
    return ts, o, h, l, c, v


def check_signal(df5, df1, order_book=None):
    """('long'|'short', atr1, close1) или None. df5/df1 — сырые ohlcv.
    order_book=None → OB-фильтр пропускается, его делает вызывающий по реальному fetch."""
    try:
        _, _, h5, _, c5, _ = _cols(df5)
        ts1, _, h1, l1, c1, v1 = _cols(df1)
    except Exception:
        return None
    if c1.size < max(BB_N, ATR_N + 1, VOL_SMA_N + 1, BREAK_N + 1, RSI_N + 1):
        return None
    if c5.size < 21:
        return None

    close = float(c1[-1])
    atr1 = atr_last(h1, l1, c1, ATR_N)
    if not atr1 or close <= 0:
        return None
    if (atr1 / close) * 100.0 < ATR_MIN_PCT:
        return None

    ema5 = ema_last(c5, 21)
    if ema5 is None:
        return None

    bb = bollinger_last(c1, BB_N, BB_K)
    rsi = rsi_last(c1, RSI_N)
    vol_base = vol_sma_prev(v1, VOL_SMA_N)
    if bb is None or rsi is None or not vol_base or vol_base <= 0:
        return None
    upper, _, lower = bb
    vol_ok = v1[-1] >= VOL_MULT * vol_base
    if not vol_ok:
        return None

    imb = 1.0
    has_ob = False
    if order_book:
        bids = order_book.get('bids') or []
        asks = order_book.get('asks') or []
        if not bids or not asks:
            return None
        imb = ob_imbalance(bids, asks, OB_DEPTH)
        if imb is None or imb <= 0:
            return None
        has_ob = True

    hi_prev = breakout_high(h1, BREAK_N)
    lo_prev = breakout_low(l1, BREAK_N)
    if hi_prev is None or lo_prev is None:
        return None

    if c5[-1] > ema5:
        long_trig = close > upper or close > hi_prev
        long_ob = (not has_ob) or imb >= OB_IMB
        if long_trig and RSI_LONG[0] <= rsi <= RSI_LONG[1] and long_ob:
            return 'long', atr1, close
    elif c5[-1] < ema5:
        short_trig = close < lower or close < lo_prev
        short_ob = (not has_ob) or (1.0 / imb) >= OB_IMB
        if short_trig and RSI_SHORT[0] <= rsi <= RSI_SHORT[1] and short_ob:
            return 'short', atr1, close
    return None


def market_regime(df1):
    """Справочный режим рынка (для логов): 'impulse'/'flat' по ATR-фильтру."""
    _, _, h, l, c, _ = _cols(df1)
    a = atr_last(h, l, c, ATR_N)
    if not a or c.size == 0 or c[-1] <= 0:
        return 'unknown'
    return 'impulse' if (a / c[-1]) * 100.0 >= ATR_MIN_PCT else 'flat'
