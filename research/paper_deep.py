"""research/paper_deep.py — живая research-paper обкатка кандидатов из deep_sym (раунд 4).

Кандидаты (выбор только по IS, OOS-подтверждение — см. LOG.md):
  №1 APTUSDT 15m mr_rsi5  exit tf_trail  (IS +15.2R/154, OOS +16.5R/83, 3/3 мес)
  №2 XRPUSDT 1h  mr_vwap  exit m1_trail (IS +11.4R/107, OOS +16.4R/56, 3/3 мес)

Модель исполнения 1-в-1 из lab.sim(): postOnly limit на open(следующего TF-бара)
с буфером EPS (=3bp), fill только если бар коснулся цены; SL-first при касании,
trail/BE обновляется по ЗАКРЫТИЮ TF-бара (как в sim), cap 361 TF-бар, комиссии
maker-in 2bp + taker-out 5.5bp (tp -> maker).
Каждая закрытая сделка сравнивается с тенью бэктеста: lab.sim() на том же окне
данных в сигнальном индексе — расхождение live vs shadow = сигнал деградации модели.

Демон:  ./venv/bin/python research/paper_deep.py            (цикл 60с)
Разово: ./venv/bin/python research/paper_deep.py --once
"""
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import research.lab as lab
import research.deep as deep
from research.data import _fetch_pages

STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'results', 'paper_deep.json')
CYCLE_SEC = 60
LOOKBACK_BARS = 8500
EPS = lab.EPS
MAKER, TAKER = lab.MAKER, lab.TAKER
MASK_CFG = dict(atr_min_pct=0.03, vol_mult=1.3)

CANDIDATES = [
    dict(key='APT_mr_rsi5', symbol='APTUSDT', tf_minutes=15, idea='mr_rsi5',
         exit=deep.DEEP_EXITS['tf_trail'],
         is_sum_R=15.21, oos_sum_R=16.47),
    dict(key='XRP_mr_vwap', symbol='XRPUSDT', tf_minutes=60, idea='mr_vwap',
         exit=deep.DEEP_EXITS['m1_trail'],
         is_sum_R=11.44, oos_sum_R=16.39),
]


def now_ms():
    return int(time.time() * 1000)


def tf_ms(minutes):
    return minutes * 60000


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return dict(trades=[], cand={c['key']: {} for c in CANDIDATES})


def save_state(st):
    tmp = STATE_PATH + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(st, f, ensure_ascii=False, indent=1)
    os.replace(tmp, STATE_PATH)


def build_lab(symbol, bars, minutes):
    rows = lab.to_tf([(int(b[0]), b[1], b[2], b[3], b[4], b[5]) for b in bars], minutes)
    return lab.LabData(symbol, rows)


def closed_index(ld, minutes, ts_now):
    """Индекс последнего ЗАКРЫТОГО TF-бара (или -1)."""
    period = tf_ms(minutes)
    idx = np.flatnonzero(ld.ts + period <= ts_now)
    return int(idx[-1]) if idx.size else -1


def live_R(side, sl_d, entry, exit_px, reason):
    """R и fees в долях риска — формулы 1-в-1 из lab.sim()."""
    dirn = 1.0 if side == 'long' else -1.0
    fee_out = MAKER if reason == 'tp' else TAKER
    fpn = MAKER + fee_out * exit_px / entry
    pnl = dirn * (exit_px - entry) - entry * fpn
    return round(float(pnl / sl_d), 4), round(float(fpn / (sl_d / entry)), 3)


def iso(ms):
    return datetime.datetime.fromtimestamp(ms / 1000, datetime.timezone.utc) \
        .strftime('%m-%d %H:%M')


def close_trade(c, st, pos, exit_px, reason, ts):
    entry = pos['entry']
    R, fees_r = live_R(pos['side'], pos['sl_d'], entry, exit_px, reason)
    t = dict(key=c['key'], symbol=c['symbol'], side=pos['side'],
             signal_ts=pos['signal_ts'], entry_ts=pos['entry_ts'], exit_ts=ts,
             entry=round(float(entry), 8), exit=round(float(exit_px), 8),
             exit_reason=reason, sl_pct=round(pos['sl_d'] / entry, 6),
             R=R, fees_R=fees_r)
    try:
        bars = _fetch_pages(c['symbol'], ts + 60000, LOOKBACK_BARS)
        ld = build_lab(c['symbol'], bars, c['tf_minutes'])
        idx = np.flatnonzero(ld.ts == pos['signal_ts'])
        if idx.size:
            sh = lab.sim(ld, c['exit'], int(idx[0]), pos['side'], None)
            if sh:
                t['shadow'] = dict(R=sh['R'], exit_reason=sh['exit_reason'],
                                   entry=sh['entry'], ts_exit=sh['ts_exit'])
    except Exception as e:
        print('[close:%s] shadow fail: %s' % (c['key'], e), flush=True)
    st['trades'].append(t)
    agg = st['cand'][c['key']].setdefault(
        'agg', dict(n=0, sum_R=0.0, wins=0, sum_R_shadow=0.0, n_shadow=0))
    agg['n'] += 1
    agg['sum_R'] = round(agg['sum_R'] + R, 4)
    agg['wins'] += int(R > 0)
    if t.get('shadow'):
        agg['n_shadow'] += 1
        agg['sum_R_shadow'] = round(agg['sum_R_shadow'] + t['shadow']['R'], 4)
    print('[%s] CLOSE %s %.6f->%.6f R=%+.3f (%s) fees_R=%.2f shadow=%s' % (
        c['key'], pos['side'], entry, exit_px, R, reason, fees_r,
        ('%+.3f/%s' % (t['shadow']['R'], t['shadow']['exit_reason']))
        if t.get('shadow') else 'n/a'), flush=True)


def make_position(pending, b, bucket, rr):
    entry = round(float(pending['entry']), 8)
    sl_d = round(float(pending['sl_d']), 8)
    tp = (entry + rr * sl_d) if pending['side'] == 'long' else (entry - rr * sl_d)
    return dict(side=pending['side'], entry=entry, sl_d=sl_d,
                sl=round(float(pending['sl']), 8), tp=round(tp, 8),
                atr1=float(pending['atr1']), entry_ts=int(b[0]),
                signal_ts=pending['signal_ts'], tf_open_ts=bucket, tf_bucket=bucket,
                tf_h=float(b[2]), tf_l=float(b[3]), tf_last_close=float(b[4]),
                tf_last_ts=int(b[0]), tf_done=0, trail=None)


def advance_position(c, st, pos, b, period, trail_mult, be_trigger):
    """Один 1m-бар открытой позиции. None — позиция закрылась.

    SL-first по 1m-бару с trail последнего закрытого TF-бара; trail/BE обновляется
    на закрытии TF-бара по его полным h/l (как в lab.sim); cap 361 TF-бар по close.
    """
    ts, o, h, l_c, cl = int(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4])
    bucket = ts // period * period
    side = pos['side']
    sl_now = pos['trail'] if pos.get('trail') is not None else pos['sl']
    hit_sl = (l_c <= sl_now) if side == 'long' else (h >= sl_now)
    hit_tp = (h >= pos['tp']) if side == 'long' else (l_c <= pos['tp'])
    if hit_sl:
        px = min(o, sl_now) if side == 'long' else max(o, sl_now)
        reason = 'trail' if pos.get('trail') is not None else 'sl'
        close_trade(c, st, pos, px, reason, ts)
        return None
    if hit_tp:
        px = max(o, pos['tp']) if side == 'long' else min(o, pos['tp'])
        close_trade(c, st, pos, px, 'tp', ts)
        return None
    if bucket != pos['tf_bucket']:
        pos['tf_done'] = int(pos.get('tf_done', 0)) + 1
        if pos['tf_done'] >= 361:
            close_trade(c, st, pos, pos['tf_last_close'], 'cap', pos['tf_last_ts'])
            return None
        entry, sl_d, atr1 = pos['entry'], pos['sl_d'], pos['atr1']
        tf_h, tf_l = pos.get('tf_h'), pos.get('tf_l')
        if tf_h is not None:
            profit_now = (tf_h - entry) if side == 'long' else (entry - tf_l)
            trail = pos.get('trail')
            if trail is None:
                if profit_now >= be_trigger * sl_d:
                    trail = (tf_h - trail_mult * atr1) if side == 'long' \
                        else (tf_l + trail_mult * atr1)
            else:
                trail = max(trail, tf_h - trail_mult * atr1) if side == 'long' \
                    else min(trail, tf_l + trail_mult * atr1)
            if trail is not None:
                pos['trail'] = round(float(trail), 8)
        pos['tf_bucket'] = bucket
        pos['tf_h'], pos['tf_l'] = h, l_c
        pos['tf_last_close'], pos['tf_last_ts'] = cl, ts
    else:
        pos['tf_h'] = max(pos.get('tf_h', h), h)
        pos['tf_l'] = min(pos.get('tf_l', l_c), l_c)
        pos['tf_last_close'], pos['tf_last_ts'] = cl, ts
    return pos


def process_candidate(c, bars, ts_now, st):
    """Один цикл кандидата: доводка позиции/ордера по новым 1m-барам, затем
    проверка сигнала на последнем закрытом TF-баре."""
    minutes = c['tf_minutes']
    period = tf_ms(minutes)
    cfg = c['exit']
    rr = cfg.get('rr', 2.5)
    trail_mult = cfg.get('trail_atr', 2.0)
    be_trigger = cfg.get('be_trigger', 1.0)
    cs = st['cand'].setdefault(c['key'], {})
    last_bar_ts = int(cs.get('last_bar_ts') or 0)
    last_sig_ts = int(cs.get('last_sig_ts') or 0)

    for b in bars:
        ts = int(b[0])
        if ts <= last_bar_ts:
            continue
        bucket = ts // period * period
        pos = cs.get('pos')
        pending = cs.get('pending')
        if pos:
            cs['pos'] = advance_position(c, st, pos, b, period, trail_mult, be_trigger)
        elif pending:
            if ts >= pending['valid_until']:
                cs['pending'] = None
                print('[%s] -- ордер не исполнился (TF закрылся без касания), sig %s'
                      % (c['key'], iso(pending['signal_ts'])), flush=True)
            else:
                entry = pending['entry']
                touched = (b[3] <= entry) if pending['side'] == 'long' else (b[2] >= entry)
                if touched:
                    pos = make_position(pending, b, bucket, rr)
                    cs['pos'] = pos
                    cs['pending'] = None
                    print('[%s] FILL %s @ %.6f (sl %.6f) sig %s' % (
                        c['key'], pos['side'], entry, pos['sl'],
                        iso(pos['signal_ts'])), flush=True)
                    cs['pos'] = advance_position(c, st, pos, b, period,
                                                 trail_mult, be_trigger)
        last_bar_ts = ts
    cs['last_bar_ts'] = last_bar_ts

    if cs.get('pos') or cs.get('pending') or not bars:
        return
    ld = build_lab(c['symbol'], bars, minutes)
    M = lab.masks(ld, MASK_CFG)
    la, sa = M[c['idea']]
    i_close = closed_index(ld, minutes, ts_now)
    if i_close < 0:
        return
    sig_ts = int(ld.ts[i_close])
    if sig_ts <= last_sig_ts:
        return
    cs['last_sig_ts'] = sig_ts
    long_sig, short_sig = bool(la[i_close]), bool(sa[i_close])
    if not (long_sig or short_sig):
        return
    side = 'long' if long_sig else 'short'
    i_entry = i_close + 1
    if i_entry >= ld.ts.size:
        return
    o_next = float(ld.o[i_entry])
    entry = o_next * (1 - EPS) if side == 'long' else o_next * (1 + EPS)
    sl_d = lab._sl_dist_generic(cfg, float(ld.c[i_close]), float(ld.atr[i_close]))
    sl = entry - sl_d if side == 'long' else entry + sl_d
    cs['pending'] = dict(side=side, entry=round(entry, 8), sl_d=round(sl_d, 8),
                         sl=round(sl, 8), atr1=float(ld.atr[i_close]),
                         signal_ts=sig_ts,
                         entry_window_ts=int(ld.ts[i_entry]),
                         valid_until=int(ld.ts[i_entry]) + period)
    print('[%s] SIGNAL %s %s bar %s -> entry %.6f sl %.6f' % (
        c['key'], side, c['idea'], iso(sig_ts), entry, sl), flush=True)


DAY_MS = 86400000


def cycle(st):
    ts_now = now_ms()
    for c in CANDIDATES:
        try:
            bars = _fetch_pages(c['symbol'], ts_now, LOOKBACK_BARS)
            if len(bars) < 300:
                print('[%s] мало баров (%d), пропуск' % (c['key'], len(bars)), flush=True)
                continue
            process_candidate(c, bars, ts_now, st)
        except Exception as e:
            print('[%s] cycle error: %s' % (c['key'], e), flush=True)
    st['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for c in CANDIDATES:
        cs = st['cand'].get(c['key'], {})
        agg = cs.get('agg')
        if agg:
            cs['summary'] = dict(
                n=agg['n'], sum_R=agg['sum_R'],
                win_rate=round(100.0 * agg['wins'] / agg['n'], 1) if agg['n'] else 0.0,
                sum_R_shadow=agg['sum_R_shadow'], n_shadow=agg['n_shadow'],
                open=bool(cs.get('pos')), pending=bool(cs.get('pending')))
    save_state(st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--once', action='store_true')
    args = ap.parse_args()
    st = load_state()
    print('[paper_deep] старт: %d кандидатов, state=%s' % (len(CANDIDATES), STATE_PATH),
          flush=True)
    while True:
        cycle(st)
        if args.once:
            break
        time.sleep(CYCLE_SEC)


if __name__ == '__main__':
    main()
