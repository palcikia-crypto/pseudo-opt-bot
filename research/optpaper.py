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

РАСШИРЕНИЕ (раунд 9e): группа 'exp' — 5 быстрых вариантов с ослабленным фильтром
(atr_override 0.02–0.025 vs 0.03, экспирации 30/60). ИЗОЛИРОВАННЫЙ счёт: у группы
exp своя equity и свой HWM («своя касса»). ЛЕСТНИЦА АВТО-ОТКАТА (без участия
человека): exp −8% от пика → режим core (расширение выключено, как до 9e);
exp −15% → режим paused (новых входов нет нигде); core −15% → paused.
Открытые контракты при откате ДОРАБАТЫВАЮТ до экспирации (резких отмен нет).

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
         expiry_min=60, grp='core',
         status='VAL-подтверждён (optevo: wr_dig 65.2%)'),
    dict(key='XRP_digital', symbol='XRPUSDT', tf_minutes=60, idea='mr_vwap',
         expiry_min=120, grp='core', status='исследование (optevo seed)'),
    dict(key='SOL_digital', symbol='SOLUSDT', tf_minutes=60, idea='mr_vwap',
         expiry_min=120, grp='core', status='исследование (optevo seed)'),
    dict(key='SEI_digital', symbol='SEIUSDT', tf_minutes=15,
         idea='micro2:doji_exhaust', expiry_min=30, grp='core',
         status='исследование (optevo seed)'),
    dict(key='BTC_digital', symbol='BTCUSDT', tf_minutes=15, idea='impulse',
         expiry_min=30, grp='core', status='исследование (optevo seed)'),
]

# Расширение (раунд 9e): ослабленные фильтры → сигналы чаще. Изолированный
# счёт: своя equity, своё HWM. При просадке группы −8% от максимума —
# АВТО-ОТКАТ на core (как до расширения), при −15% — полная пауза новых входов.
EXPANDED = [
    dict(key='APT_fast', symbol='APTUSDT', tf_minutes=15, idea='mr_rsi5',
         expiry_min=60, grp='exp', atr_override=0.025,
         status='расширение: APT mr_rsi5, фильтр слабее (частота выше)'),
    dict(key='XRP_fast', symbol='XRPUSDT', tf_minutes=60, idea='mr_vwap',
         expiry_min=60, grp='exp', atr_override=0.025,
         status='расширение: XRP mr_vwap exp60 (короче/чаще)'),
    dict(key='SOL_fast', symbol='SOLUSDT', tf_minutes=60, idea='mr_vwap',
         expiry_min=60, grp='exp', atr_override=0.025,
         status='расширение: SOL mr_vwap exp60'),
    dict(key='BTC_fast', symbol='BTCUSDT', tf_minutes=15, idea='impulse',
         expiry_min=30, grp='exp', atr_override=0.02,
         status='расширение: BTC impulse, фильтр слабее'),
    dict(key='SEI_fast', symbol='SEIUSDT', tf_minutes=15,
         idea='micro2:doji_exhaust', expiry_min=30, grp='exp',
         status='расширение: SEI doji_exhaust exp30'),
]

# Границы автовозврата
EXP_REVERT_DDN = 0.92   # equity_exp упала до 92% от своего максимума → возврат к core
EXP_PAUSE_DDN = 0.85    # 85% → пауза новых входов (всё уже закрыто/ждёт)
CORE_PAUSE_DDN = 0.85   # ядро просело на 15% → пауза всего (сигнал беды)


def now_ms():
    return int(time.time() * 1000)


def build_labs(symbol, bars, minutes):
    rows1 = [(int(b[0]), b[1], b[2], b[3], b[4], b[5]) for b in bars]
    ld1 = lab.LabData(symbol, rows1)
    ldtf = lab.LabData(symbol, lab.to_tf(rows1, minutes))
    return ld1, ldtf


def masks_for_idea(ldtf, idea, cfg=None):
    import research.micro2 as micro2
    if idea.startswith('micro2:'):
        return micro2.micro2_families(ldtf, cfg or MASK_CFG)[idea.split(':', 1)[1]]
    return lab.masks(ldtf, cfg or MASK_CFG)[idea]


def cand_cfg(c):
    """Пер-кандидатный фильтр: atr_override слабее дефолтного 0.03 у exp-группы."""
    if 'atr_override' not in c:
        return MASK_CFG
    cfg = dict(MASK_CFG)
    cfg['atr_min_pct'] = c['atr_override']
    return cfg


def ALL_CANDIDATES():
    return CANDIDATES + EXPANDED


def _utc(now=True):
    return datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%S')


def _notice(st, msg):
    st.setdefault('notices', [])
    st['notices'].append(dict(ts=_utc(), msg=msg))
    st['notices'] = st['notices'][-20:]
    print('[watch safeguard] ' + msg, flush=True)


def _guards(st, grp):
    """Лестница авто-отката — вызывается после каждой закрытой сделки группы."""
    if grp == 'exp':
        if st['mode'] == 'expanded' and \
                st['equity_exp'] <= st['hwm_exp'] * EXP_REVERT_DDN:
            st['mode'] = 'core'
            _notice(st, 'АВТО-ОТКАТ: расширение отключено (exp-касса %s просела >8%% '
                         'от пика %s), торгуем как раньше — только ядро'
                    % (round(st['equity_exp'], 2), round(st['hwm_exp'], 2)))
            return
        if st['mode'] == 'core' and \
                st['equity_exp'] <= st['hwm_exp'] * EXP_PAUSE_DDN:
            st['mode'] = 'paused'
            _notice(st, 'ПАУЗА: exp-касса просела >15% от пика — все новые входы '
                         'опционов остановлены')
            return
    if grp == 'core' and st['mode'] != 'paused' and \
            st['equity_core'] <= st['hwm_core'] * CORE_PAUSE_DDN:
        st['mode'] = 'paused'
        _notice(st, 'ПАУЗА: ядро просело >15% от пика — новые входы остановлены')


def sigma_1m(ld1):
    r = np.asarray(ld1.ret1, dtype=float)
    r = r[np.isfinite(r)]
    return float(np.std(r)) if r.size > 500 else None


def load_state():
    st = None
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH) as f:
                st = json.load(f)
        except Exception as e:
            print('[optpaper] state unreadable, fresh: %s' % e, flush=True)
            st = None
    if st is None:
        st = dict(trades=[], cand={}, updated='')
    # --- миграция 9e: изолированные кассы core/exp, HWM, режим, уведомления ---
    if 'mode' not in st or st['mode'] not in ('expanded', 'core', 'paused'):
        # старый режим был строкой-описанием → переименовываем в режим
        st['mode_label'] = st.get(
            'mode', 'виртуальный бинарный опцион (paper): payout +85%/-100%, '
                    'экспирация фикс; премия модельная k·σ√E; НЕ биржевой опцион')
        st['mode'] = 'expanded'
    st.setdefault('mode_label', 'виртуальный бинарный опцион (paper): payout '
                                '+85%/-100%, экспирация фикс; НЕ биржевой опцион')
    st.setdefault('data', 'цены Bybit USDT-perp 1m (fetchOHLCV), TF-агрегация '
                          '1-в-1 с бэком')
    st.setdefault('cand', {})
    st.setdefault('trades', [])
    st.setdefault('notices', [])
    st.setdefault('_paused_log_ts', 0)
    legacy_eq = float(st.get('equity', 1000.0))
    st.setdefault('equity_core', legacy_eq)   # старое equity = касса ядра
    st.setdefault('equity_exp', 1000.0)       # расширению — новая касса $1000
    st.setdefault('hwm_core', st['equity_core'])
    st.setdefault('hwm_exp', st['equity_exp'])
    st['equity'] = round(st['equity_core'] + st['equity_exp'], 4)  # legacy-сумма
    for c in ALL_CANDIDATES():
        st['cand'].setdefault(c['key'], dict(n=0, wins=0, pnl=0.0))
    return st


def save_state(st):
    tmp = STATE_PATH + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(st, f, ensure_ascii=False, indent=1)
    os.replace(tmp, STATE_PATH)


def close_option(c, st, pos, exit_px):
    grp = c.get('grp', 'core')
    eq_key = 'equity_exp' if grp == 'exp' else 'equity_core'
    entry = pos['entry']
    dirn = 1.0 if pos['side'] == 'long' else -1.0
    mv = dirn * (exit_px / entry - 1.0)
    win = mv > 0
    stake = pos['stake']
    pnl = round(stake * (PAYOUT if win else -1.0), 4)
    st[eq_key] = round(st[eq_key] + pnl, 4)
    st['hwm_' + ('exp' if grp == 'exp' else 'core')] = \
        max(st['hwm_' + ('exp' if grp == 'exp' else 'core')], st[eq_key])
    st['equity'] = round(st['equity_core'] + st['equity_exp'], 4)
    t = dict(key=c['key'], symbol=c['symbol'], side=pos['side'],
             idea=c['idea'], tf=c['tf_minutes'], expiry_min=c['expiry_min'],
             entry_ts=pos['entry_ts'], expiry_ts=pos['expiry_ts'],
             exit_ts=now_ms(), entry=round(float(entry), 8),
             exit=round(float(exit_px), 8),
             move_bp=round(mv * 1e4, 1), prem_bp=round(pos['prem_bp'], 1),
             win=bool(win), stake=stake, pnl=pnl, grp=grp,
             vanilla_bp=round(max(0.0, mv) * 1e4 - pos['prem_bp'], 1))
    st['trades'].append(t)
    a = st['cand'][c['key']]
    a['n'] += 1
    a['wins'] += int(win)
    a['pnl'] = round(a['pnl'] + pnl, 4)
    print('[%s] CLOSE %s exp=%dm %s move=%.1fbp prem=%.1fbp pnl=%+.2f eq_%s=%.2f'
          % (c['key'], pos['side'], c['expiry_min'],
             'WIN' if win else 'LOSS', mv * 1e4, pos['prem_bp'],
             pnl, grp, st[eq_key]), flush=True)
    _guards(st, grp)


def cycle_once(st):
    ts_now = now_ms()
    mode = st['mode']
    for c in ALL_CANDIDATES():
        try:
            grp = c.get('grp', 'core')
            bars = _fetch_pages(c['symbol'], ts_now + 60000, LOOKBACK_BARS)
            ld1, ldtf = build_labs(c['symbol'], bars, c['tf_minutes'])
            period = c['tf_minutes'] * 60000
            cd = st['cand'][c['key']]
            pos = cd.get('pos')
            if pos:
                # открытый контракт дорабатывает до экспирации даже при откате
                if ts_now >= pos['expiry_ts']:
                    idx = np.flatnonzero(ld1.ts < pos['expiry_ts'])
                    if idx.size:
                        close_option(c, st, pos, float(ld1.c[idx[-1]]))
                        cd.pop('pos')
                        mode = st['mode']  # после закрытия режим мог смениться
                continue
            # --- новые входы: ворота режима ---
            if st['mode'] == 'paused' or (grp == 'exp' and st['mode'] != 'expanded'):
                if st['mode'] == 'paused' and \
                        ts_now - st.get('_paused_log_ts', 0) > 3600000:
                    st['_paused_log_ts'] = ts_now
                    print('[watch safeguard] режим PAUSED — новые входы остановлены '
                          '(ворота раунда 9e)', flush=True)
                continue
            closed = np.flatnonzero(ldtf.ts + period <= ts_now)
            if not closed.size:
                continue
            i = int(closed[-1])
            if int(ldtf.ts[i]) == cd.get('last_sig_ts'):
                continue
            larr, sarr = masks_for_idea(ldtf, c['idea'], cand_cfg(c))
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
                eq_grp = st['equity_exp' if grp == 'exp' else 'equity_core']
                pos = dict(side='long' if larr[i] else 'short',
                           entry=entry, entry_ts=int(ld1.ts[j]),
                           expiry_ts=int(ld1.ts[j]) + c['expiry_min'] * 60000,
                           prem=prem, prem_bp=prem * 1e4,
                           stake=round(eq_grp * 0.01, 4),
                           signal_ts=sig)
                cd['pos'] = pos
                cd['last_sig_ts'] = sig
                print('[%s] OPEN %s @%.6f exp=%dm stake=%.2f prem=%.1fbp [%s]'
                      % (c['key'], pos['side'], entry, c['expiry_min'],
                         pos['stake'], pos['prem_bp'], grp),
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
        cinfo = {c['key']: c for c in ALL_CANDIDATES()}
        meta['optpaper'] = dict(
            updated=st['updated'], equity=st['equity'],
            equity_core=st['equity_core'], equity_exp=st['equity_exp'],
            regime=st['mode'], mode=st.get('mode_label', ''),
            notices=st.get('notices', [])[-5:], data=st['data'],
            cands={k: dict(n=v['n'],
                           wr=round(100.0 * v['wins'] / v['n'], 1) if v['n'] else None,
                           pnl=v['pnl'], open_pos=bool(v.get('pos')),
                           grp=cinfo.get(k, {}).get('grp', 'core'),
                           status=cinfo.get(k, {}).get('status', ''))
                   for k, v in st['cand'].items()})
        json.dump(meta, open(pm, 'w'), indent=1, ensure_ascii=False)
        save_state(st)
        if args.once:
            break
        time.sleep(CYCLE_SEC)


if __name__ == '__main__':
    main()
