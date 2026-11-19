"""research/indicators_vec.py — векторные (по всему массиву) версии индикаторов
для быстрой предфильтрации баров перед точным check_signal()."""
import numpy as np


def wilder_atr(h, l, c, n=14):
    """ATR (Wilder) по всей серии; atr[i] использует TR баров [i-n+1..i]."""
    h, l, c = np.asarray(h), np.asarray(l), np.asarray(c)
    pc = c[:-1]
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - pc), np.abs(l[1:] - pc)))
    atr = np.full(c.size, np.nan)
    if tr.size < n:
        return atr
    a = tr[:n].mean()
    atr[n] = a
    for i in range(n, tr.size):
        a = (a * (n - 1) + tr[i]) / n
        atr[i + 1] = a
    return atr


def wilder_rsi(c, n=14):
    c = np.asarray(c, dtype=float)
    d = np.diff(c)
    up = np.clip(d, 0, None)
    dn = np.clip(-d, 0, None)
    rsi = np.full(c.size, np.nan)
    if d.size < n:
        return rsi
    au, ad = up[:n].mean(), dn[:n].mean()
    rsi[n] = 100.0 if ad == 0 else 100.0 - 100.0 / (1.0 + au / ad)
    for i in range(n, d.size):
        au = (au * (n - 1) + up[i]) / n
        ad = (ad * (n - 1) + dn[i]) / n
        rsi[i + 1] = 100.0 if ad == 0 else 100.0 - 100.0 / (1.0 + au / ad)
    return rsi


def rolling_mean(x, n, shift=0):
    """shift=1: среднее по n предыдущим барам (текущий не входит)."""
    x = np.asarray(x, dtype=float)
    cs = np.concatenate(([0.0], np.cumsum(x)))
    out = np.full(x.size, np.nan)
    for i in range(n + shift - 1, x.size):
        out[i] = (cs[i - shift + 1] - cs[i - shift + 1 - n]) / n
    return out


def rolling_std(x, n):
    x = np.asarray(x, dtype=float)
    cs = np.concatenate(([0.0], np.cumsum(x)))
    cs2 = np.concatenate(([0.0], np.cumsum(x * x)))
    out = np.full(x.size, np.nan)
    for i in range(n - 1, x.size):
        m = (cs[i + 1] - cs[i + 1 - n]) / n
        v = (cs2[i + 1] - cs2[i + 1 - n]) / n - m * m
        out[i] = np.sqrt(max(v, 0.0))
    return out


def rollmax_prev(h, n):
    """max по n барам, предшествующим текущему (без него)."""
    h = np.asarray(h, dtype=float)
    out = np.full(h.size, np.nan)
    for i in range(n, h.size):
        out[i] = h[i - n:i].max()
    return out


def rollmin_prev(l, n):
    l = np.asarray(l, dtype=float)
    out = np.full(l.size, np.nan)
    for i in range(n, l.size):
        out[i] = l[i - n:i].min()
    return out


def ema_series(x, n):
    """EMA(n) по всей серии ( первые n-1 — nan)."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.size, np.nan)
    if x.size < n:
        return out
    alpha = 2.0 / (n + 1.0)
    e = x[:n].mean()
    out[n - 1] = e
    for i in range(n, x.size):
        e = alpha * x[i] + (1.0 - alpha) * e
        out[i] = e
    return out
