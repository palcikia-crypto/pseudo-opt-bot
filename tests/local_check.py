"""local_check.py — офлайн-санити логики без сети: индикаторы, стратегия, риск, лот."""
import numpy as np

import config as C
import indicators as I
import risk_manager as RM
from strategy import check_signal


def synth(n=200, seed=7, drift=0.0004, spike_at=None):
    rng = np.random.default_rng(seed)
    base = 100.0
    closes = [base]
    highs = [base * 1.001]
    lows = [base * 0.999]
    for i in range(1, n):
        step = drift + rng.normal(0, 0.0012)
        if spike_at is not None and i >= spike_at:
            step = drift * 6 + 0.004
        c = closes[-1] * (1 + step)
        closes.append(c)
        highs.append(max(c, closes[-2]) * (1 + rng.normal(0, 0.0004)))
        lows.append(min(c, closes[-2]) * (1 - rng.normal(0, 0.0004)))
    closes = np.array(closes)
    highs = np.array(highs)
    lows = np.array(lows)
    vols = np.abs(rng.normal(500, 120, n))
    vols[-1] = vols.mean() * 3.0
    ohlcv1 = [[i * 60000, closes[i-1] if i else base, highs[i], lows[i], closes[i], vols[i]]
              for i in range(n)]
    ohlcv5 = [[i * 300000, closes[i-1] if i else base, highs[i], lows[i], closes[i], vols[i]]
              for i in range(n//5)]
    return ohlcv5, ohlcv1


def main():
    ohlcv5, ohlcv1 = synth()
    c1 = [r[4] for r in ohlcv1]
    h1 = [r[2] for r in ohlcv1]
    l1 = [r[3] for r in ohlcv1]

    e = I.ema_last(c1, 21)
    bb = I.bollinger_last(c1, 20, 2.0)
    r = I.rsi_last(c1, 14)
    a = I.atr_last(h1, l1, c1, 14)
    assert e and bb and 0 <= r <= 100 and a > 0, (e, bb, r, a)

    close = c1[-1]
    d = RM.sl_distance(close, a)
    assert close * C.SL_MIN_PCT - 1e-12 <= d <= close * C.SL_MAX_PCT + 1e-12, d
    q = RM.qty_for_close(10.0, d)
    assert abs(q * d - 10.0) < 1e-9, q
    assert RM.margin_ok(q, close, 1000.0)

    ob = {'bids': [[close * 0.999, 50]] * 5, 'asks': [[close * 1.001, 50]] * 5}
    imb = I.ob_imbalance(ob['bids'], ob['asks'], 5)
    assert abs(imb - 1.0) < 1e-9, imb
    ob2 = {'bids': [[close * 0.999, 100]] * 5, 'asks': [[close * 1.001, 50]] * 5}
    assert abs(I.ob_imbalance(ob2['bids'], ob2['asks'], 5) - 2.0) < 1e-9

    sig = check_signal(ohlcv5, ohlcv1, ob)
    print('сигнал на синтетике:', sig, '| ATR%%: %.3f' % ((a / close) * 100))

    flat1 = [[r0[0], r0[1], r0[2], r0[3], close * 1.0, 100.0] for r0 in ohlcv1]
    sig2 = check_signal(ohlcv5, flat1, ob)
    assert sig2 is None, sig2
    print('SANITY OK: индикаторы, границы SL, лот, ATR-фильтр, флэт-отсев')


if __name__ == '__main__':
    main()
