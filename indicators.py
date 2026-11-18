"""indicators.py — быстрые numpy-расчёты индикаторов на массивах OHLCV."""
import numpy as np


def ema_last(x, n):
    x = np.asarray(x, dtype=float)
    if x.size < n:
        return None
    alpha = 2.0 / (n + 1.0)
    e = x[:n].mean()
    for v in x[n:]:
        e = alpha * v + (1.0 - alpha) * e
    return float(e)


def bollinger_last(x, n, k):
    x = np.asarray(x, dtype=float)
    if x.size < n:
        return None
    win = x[-n:]
    mid = float(win.mean())
    sd = float(win.std(ddof=0))
    return mid + k * sd, mid, mid - k * sd


def rsi_last(x, n):
    x = np.asarray(x, dtype=float)
    if x.size < n + 1:
        return None
    d = np.diff(x)
    up = d.clip(min=0.0)
    dn = (-d).clip(min=0.0)
    au = up[:n].mean()
    ad = dn[:n].mean()
    for i in range(n, d.size):
        au = (au * (n - 1) + up[i]) / n
        ad = (ad * (n - 1) + dn[i]) / n
    if ad == 0.0:
        return 100.0
    rs = au / ad
    return float(100.0 - 100.0 / (1.0 + rs))


def atr_last(high, low, close, n):
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    if high.size < n + 1:
        return None
    prev_close = close[:-1]
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - prev_close),
                                                   np.abs(low[1:] - prev_close)))
    a = tr[:n].mean()
    for i in range(n, tr.size):
        a = (a * (n - 1) + tr[i]) / n
    return float(a)


def vol_sma_prev(vols, n):
    v = np.asarray(vols, dtype=float)
    if v.size < n + 1:
        return None
    return float(v[-(n + 1):-1].mean())


def breakout_high(high, lookback_n):
    h = np.asarray(high, dtype=float)
    if h.size < lookback_n + 1:
        return None
    return float(h[-(lookback_n + 1):-1].max())


def breakout_low(low, lookback_n):
    l = np.asarray(low, dtype=float)
    if l.size < lookback_n + 1:
        return None
    return float(l[-(lookback_n + 1):-1].min())


def ob_imbalance(bids, asks, depth):
    b = bids[:depth]
    a = asks[:depth]
    bid_sum = sum(float(x[1]) for x in b)
    ask_sum = sum(float(x[1]) for x in a)
    if ask_sum <= 0.0:
        return None
    return bid_sum / ask_sum
