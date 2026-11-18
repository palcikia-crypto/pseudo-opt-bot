"""risk_manager.py — расчёт лота под фиксированный риск, лимиты и дневной стоп-аут."""
import json
import logging
import os
import time
import datetime

from config import (DAY_LOSS_PCT, LEVERAGE, MAX_POSITIONS, RISK_PER_TRADE_PCT,
                    RR, SL_ATR_MULT, SL_MAX_PCT, SL_MIN_PCT, STATE_FILE)

log = logging.getLogger('risk')


def sl_distance(close, atr1):
    d = SL_ATR_MULT * atr1
    d = max(d, close * SL_MIN_PCT)
    d = min(d, close * SL_MAX_PCT)
    return d


def tp_distance(sl_dist):
    return RR * sl_dist


def qty_for_close(risk_usdt, sl_dist):
    if sl_dist <= 0:
        return 0.0
    return risk_usdt / sl_dist


def margin_ok(qty, price, free_usdt):
    need = qty * price / LEVERAGE
    return need <= free_usdt * 0.95


class RiskManager:
    def __init__(self):
        self.state = self._load()

    def _load(self):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            return {}

    def _save(self):
        try:
            tmp = STATE_FILE + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(self.state, f)
            os.replace(tmp, STATE_FILE)
        except Exception as e:
            log.warning('state save: %s', e)

    def rollover(self, balance):
        today = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d')
        if self.state.get('day') != today:
            self.state = {'day': today, 'day_start_balance': balance,
                          'realized_pnl': 0.0, 'halted': False}
            self._save()
            return True
        return False

    def risk_usdt(self, balance):
        return max(0.0, balance * RISK_PER_TRADE_PCT)

    def allow_new(self, open_count):
        return (not self.state.get('halted')) and open_count < MAX_POSITIONS

    def register_close(self, pnl):
        self.state['realized_pnl'] = round(
            self.state.get('realized_pnl', 0.0) + float(pnl), 6)
        start = self.state.get('day_start_balance') or 0.0
        if start > 0 and self.state['realized_pnl'] <= -start * DAY_LOSS_PCT:
            self.state['halted'] = True
            log.warning('ДНЕВНОЙ СТОП-АУТ: pnl %s <= -%.1f%% от %s',
                        self.state['realized_pnl'], DAY_LOSS_PCT * 100, start)
        self._save()
        return self.state['halted']

    @property
    def halted(self):
        return bool(self.state.get('halted'))

    @property
    def realized_pnl(self):
        return float(self.state.get('realized_pnl', 0.0))

    @property
    def day_start_balance(self):
        return float(self.state.get('day_start_balance', 0.0))
