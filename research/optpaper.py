"""research/optpaper.py — ОТДЕЛЬНЫЙ live-paper опцион-режим (раунд 9c).

ЧТО ЭТО: виртуальный БИНАРНЫЙ опцион поверх реальных цен Bybit USDT-perp.
Это НЕ биржевой опцион: опционной цепи для альтов не существует. Премия
модельная (k × σ_1m × √E), выплата фикс: угадал направление → +85% ставки,
не угадал → −100% ставки (безубыток wr 54.05%).

ОТКУДА ДАННЫЕ: fetchOHLCV-страницы 1m Bybit (3 дня), агрегация в TF-бары
(lab.to_tf, 1-в-1 с бэктестом). Сигнал — те же семейства масок
(lab/micro2), что и в бэктесте optevo.py.

КАК ТОРГУЕТ: на закрытии TF-бара с сигналом покупаем виртуальный ATM-опцион:
вход = close сигнального бара, экспирация фикс E минут, держим до конца.
Выплата по close 1m-бара на момент экспирации.

КАНДИДАТЫ (все — paper, не реальные деньги):
  APTUSDT 15m mr_rsi5 exp60      — VAL-подтверждён (optevo: wr_dig 65.2%)
  XRPUSDT 60m mr_vwap exp120     — исследование (optevo seed)
  SOLUSDT 60m mr_vwap exp120     — исследование
  SEIUSDT 15m doji_exhaust exp30 — исследование
  BTCUSDT 15m impulse exp30      — исследование

Демон: ./venv/bin/python research/optpaper.py            (цикл 60с)
Раз:  ./venv/bin/python research/optpaper.py --once
Отчёт: research/results/optpaper.json + panel_meta['optpaper']
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
from research.data import _fetch_pages

RES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
STATE_PATH = os.path.join(RES_DIR, 'optpaper.json')
CYCLE_SEC = 60
LOOKBACK_BARS = 4320
PAYOUT = 0.85
PREM_K = 0.85
MASK_CFG = dict(atr_min_pct=0.03, vol_mult=1.3)

CANDIDATES = [
    dict(key='APT_digital', symbol='APTUSDT', tf_minutes=15, idea='mr_rsi5',
         expiry_min=60, status='VAL-подтверждён (optevo: wr_dig 65.2%)'),
    dict(key='XRP_digital', symbol='XRPUSDT', tf_minutes=60, idea='mr_vwap',
         expiry_min=120, status='исследование (optevo seed)'),
    dict(key='SOL_digital', symbol='SOLUSDT', tf_minutes=60, idea='mr_vwap',
         expiry_min=120, status='исследование (optevo seed)'),
    dict(key='SEI_digital', symbol='SEIUSDT', tf_minutes=15,
         idea='micro2:doji_exhaust', expiry_min=30,
         status='исследование (optevo seed)'),
    dict(key='BTC_digital', symbol='BTCUSDT', tf_minutes=15, idea='impulse',
         expiry_min=30, status='исследование (optevo seed)'),
]


def now_ms():
    return int(time.time() * 1000)


def build_labs(symbol, bars, minutes):
    rows1 = [(int(b[0]), b[1], b[2], b[3], b[4], b[5]) for b in bars]
    ld1 = lab.LabData(symbol, rows1)
    ldtf = lab.LabData(symbol, lab.to_tf(rows1, minutes))
    return ld1, ldtf


def masks_for_idea(ldtf, idea):
    import research.micro2 as micro2
    if idea.startswith('micro2:'):
        return micro2.micro2_families(ldtf, MASK_CFG)[idea.split(':', 1)[1]]
    return lab.masks(ldtf, MASK_CFG)[idea]


def sigma_1m(ld1):
    r = np.asarray(ld1.ret1, dtype=float)
    r = r[np.isfinite(r)]
    return float(np.std(r)) if r.size > 500 else None


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return dict(
        mode='виртуальный бинарный опцион (paper): payout +85%/-100%, '
             'экспирация фикс; премия модельная k·σ√E; НЕ биржевой опцион',
        data='цены Bybit USDT-perp 1m (fetchOHLCV), TF-агрегация 1-в-1 с бэком',
        equity=1000.0, trades=[],
        cand={c['key']: dict(n=0, wins=0, pnl=0.0) for c in CANDIDATES})


def save_state(st):
    tmp = STATE_PATH + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(st, f, ensure_ascii=False, indent=1)
    os.replace(tmp, STATE_PATH)


def close_option(c, st, pos, exit_px):
    entry = pos['entry']
    dirn = 1.0 if pos['side'] == 'long' else -1.0
    mv = dirn * (exit_px / entry - 1.0)
    win = mv > 0
    stake = pos['stake']
    pnl = round(stake * (PAYOUT if win else -1.0), 4)
    st['equity'] = round(st['equity'] + pnl, 4)
    t = dict(key=c['key'], symbol=c['symbol'], side=pos['side'],
             idea=c['idea'], tf=c['tf_minutes'], expiry_min=c['expiry_min'],
             entry_ts=pos['entry_ts'], expiry_ts=pos['expiry_ts'],
             exit_ts=now_ms(), entry=round(float(entry), 8),
             exit=round(float(exit_px), 8),
             move_bp=round(mv * 1e4, 1), prem_bp=round(pos['prem_bp'], 1),
             win=bool(win), stake=stake, pnl=pnl,
             vanilla_bp=round(max(0.0, mv) * 1e4 - pos['prem_bp'], 1))
    st['trades'].append(t)
    a = st['cand'][c['key']]
    a['n'] += 1
    a['wins'] += int(win)
    a['pnl'] = round(a['pnl'] + pnl, 4)
    print('[%s] CLOSE %s exp=%dm %s move=%.1fbp prem=%.1fbp pnl=%+.2f eq=%.2f'
          % (c['key'], pos['side'], c['expiry_min'],
             'WIN' if win else 'LOSS', mv * 1e4, pos['prem_bp'],
             pnl, st['equity']), flush=True)


def cycle_once(st):
    ts_now = now_ms()
    for c in CANDIDATES:
        try:
            bars = _fetch_pages(c['symbol'], ts_now + 60000, LOOKBACK_BARS)
            ld1, ldtf = build_labs(c['symbol'], bars, c['tf_minutes'])
            period = c['tf_minutes'] * 60000
            cd = st['cand'][c['key']]
            pos = cd.get('pos')
            if pos:
                if ts_now >= pos['expiry_ts']:
                    idx = np.flatnonzero(ld1.ts < pos['expiry_ts'])
                    if idx.size:
                        close_option(c, st, pos, float(ld1.c[idx[-1]]))
                        cd.pop('pos')
            else:
                closed = np.flatnonzero(ldtf.ts + period <= ts_now)
                if not closed.size:
                    continue
                i = int(closed[-1])
                if int(ldtf.ts[i]) == cd.get('last_sig_ts'):
                    continue
                larr, sarr = masks_for_idea(ldtf, c['idea'])
                if not (larr[i] or sarr[i]):
                    continue
                sig = int(ldtf.ts[i])
                nxt = np.flatnonzero(ld1.ts >= sig + period)
                if not nxt.size:
                    continue
                j = int(nxt[0]) - 1
                entry = float(ld1.c[j])
                if not np.isfinite(entry) or entry <= 0:
                    continue
                sg = sigma_1m(ld1)
                if not sg:
                    continue
                prem = PREM_K * sg * float(np.sqrt(c['expiry_min']))
                if cd.get('last_sig_ts') != sig:
                    pos = dict(side='long' if larr[i] else 'short',
                               entry=entry, entry_ts=int(ld1.ts[j]),
                               expiry_ts=int(ld1.ts[j]) + c['expiry_min'] * 60000,
                               prem=prem, prem_bp=prem * 1e4,
                               stake=round(st['equity'] * 0.01, 4),
                               signal_ts=sig)
                    cd['pos'] = pos
                    cd['last_sig_ts'] = sig
                    print('[%s] OPEN %s @%.6f exp=%dm stake=%.2f prem=%.1fbp (%s)'
                          % (c['key'], pos['side'], entry, c['expiry_min'],
                             pos['stake'], pos['prem_bp'], c['status']),
                          flush=True)
        except Exception as e:
            print('[%s] cycle err: %s' % (c['key'], str(e)[:120]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--once', action='store_true')
    args = ap.parse_args()
    st = load_state()
    while True:
        cycle_once(st)
        st['updated'] = datetime.datetime.now(datetime.timezone.utc) \
            .strftime('%Y-%m-%dT%H:%M:%SZ')
        pm = os.path.join(RES_DIR, 'panel_meta.json')
        meta = {}
        if os.path.exists(pm):
            try:
                meta = json.load(open(pm))
            except Exception:
                meta = {}
        cinfo = {c['key']: c for c in CANDIDATES}
        meta['optpaper'] = dict(
            updated=st['updated'], equity=st['equity'],
            mode=st['mode'], data=st['data'],
            cands={k: dict(n=v['n'],
                           wr=round(100.0 * v['wins'] / v['n'], 1) if v['n'] else None,
                           pnl=v['pnl'], open_pos=bool(v.get('pos')),
                           status=cinfo.get(k, {}).get('status', ''))
                   for k, v in st['cand'].items()})
        json.dump(meta, open(pm, 'w'), indent=1, ensure_ascii=False)
        save_state(st)
        if args.once:
            break
        time.sleep(CYCLE_SEC)


if __name__ == '__main__':
    main()
