"""trader.py — жизненный цикл сделки: postOnly вход, биржевые SL/TP, безубыток, тайм-стоп."""
import asyncio
import logging
import time

import ccxt.async_support as accxt

from config import (BE_OFFSET_PCT, BE_TRIGGER, ENTRY_TIMEOUT_S, ENTRY_TIMEOUT_S as T_OUT,
                    MAX_POSITIONS, POLL_SEC, TIME_STOP_MIN)
import risk_manager as rm

log = logging.getLogger('trader')

REF_T_OUT = T_OUT


class Trade:
    __slots__ = ('symbol', 'side', 'entry', 'qty', 'sl', 'tp', 'sl_dist',
                 'opened_ts', 'be_moved', 'entry_ms')

    def __init__(self, symbol, side, entry, qty, sl_dist, tp, entry_ms=None):
        self.symbol = symbol
        self.side = side
        self.entry = float(entry)
        self.qty = float(qty)
        self.sl_dist = float(sl_dist)
        self.sl = float(entry - sl_dist) if side == 'long' else float(entry + sl_dist)
        self.tp = tp
        self.opened_ts = time.time()
        self.entry_ms = entry_ms or int(time.time() * 1000)
        self.be_moved = False

    def profit_dist(self, price):
        return (price - self.entry) if self.side == 'long' else (self.entry - price)


class Trader:
    def __init__(self, ex, risk, notify=None):
        self.ex = ex
        self.risk = risk
        self.notify = notify or (lambda msg: None)
        self.trades = {}
        self.halt_executed = False

    # ---------- вход ----------

    async def try_enter(self, symbol, side, atr1, close, order_book):
        if symbol in self.trades:
            return False
        if not self.risk.allow_new(len(self.trades)):
            return False
        book = order_book or {}
        bids, asks = book.get('bids') or [], book.get('asks') or []
        if not bids or not asks:
            return False
        limit_px = float(bids[0][0]) if side == 'long' else float(asks[0][0])
        balance = await self.ex.free_usdt()
        if balance <= 0:
            return False
        self.risk.rollover(balance)
        risk_usdt = self.risk.risk_usdt(balance)
        sl_dist = rm.sl_distance(close, atr1)
        tp_dist = rm.tp_distance(sl_dist)
        qty_raw = rm.qty_for_close(risk_usdt, sl_dist)
        qty = await self.ex.amount_to_precision(symbol, qty_raw)
        if qty <= 0:
            log.info('%s: qty %s < мин. лот — пропуск', symbol, qty_raw)
            return False
        if not rm.margin_ok(qty, limit_px, balance):
            log.warning('%s: мало свободной маржи (нужно %.2f, free %.2f)',
                        symbol, qty * limit_px / 10.0, balance)
            return False
        sl_px = limit_px - sl_dist if side == 'long' else limit_px + sl_dist
        tp_px = limit_px + tp_dist if side == 'long' else limit_px - tp_dist
        try:
            o = await self.ex.place_post_only(symbol, side, qty, limit_px, sl_px, tp_px)
        except accxt.ExchangeError as e:
            log.info('%s: postOnly не принят (%s) — пропуск', symbol, e)
            return False
        if not o or not o.get('id'):
            return False
        res = await self.ex.wait_filled(symbol, o['id'], ENTRY_TIMEOUT_S)
        if res is None:
            await self.ex.cancel(symbol, o['id'])
            log.info('%s: %s-ордер не исполнился за %ss — отмена', symbol, side,
                     ENTRY_TIMEOUT_S)
            return False
        entry = res['price'] or limit_px
        t = Trade(symbol, side, entry, res['filled'] or qty, sl_dist, tp_px)
        self.trades[symbol] = t
        log.info('ВХОД %s %s qty=%s @%.6g | SL %.6g TP %.6g (риск $%.2f)',
                 symbol, side.upper(), t.qty, entry, t.sl, t.tp, t.qty * sl_dist)
        self.notify('Вход %s %s @%.6g, SL %.6g / TP %.6g' %
                    (symbol, side.upper(), entry, t.sl, t.tp))
        return True

    # ---------- сопровождение ----------

    async def monitor_once(self):
        positions = await self.ex.positions()
        by_sym = {}
        for p in positions:
            if p.get('symbol'):
                by_sym[p['symbol']] = p
        adopted = self._adopt(by_sym)
        for symbol, t in list(self.trades.items()):
            pos = by_sym.get(symbol)
            if pos is None:
                await self._on_closed(symbol, t)
                continue
            try:
                price = float(pos.get('markPrice') or pos.get('lastPrice') or 0)
            except Exception:
                price = 0.0
            if price <= 0:
                continue
            if not t.be_moved and t.profit_dist(price) >= BE_TRIGGER * t.sl_dist:
                new_sl = t.entry + BE_OFFSET_PCT * t.entry if t.side == 'long' \
                    else t.entry - BE_OFFSET_PCT * t.entry
                ok = await self._move_sl(symbol, new_sl)
                if ok:
                    t.be_moved = True
                    t.sl = new_sl
                    log.info('БЕЗУБЫТОК %s: SL -> %.6g', symbol, new_sl)
                    self.notify('Безубыток %s @%.6g' % (symbol, new_sl))
            if (not t.be_moved and
                    time.time() - t.opened_ts > TIME_STOP_MIN * 60.0):
                log.info('ТАЙМ-СТОП %s (%.0f мин, BE не достигнут)', symbol,
                         (time.time() - t.opened_ts) / 60.0)
                self.notify('Тайм-стоп %s' % symbol)
                await self._timeout_close(symbol, t)
        return len(self.trades)

    def _adopt(self, by_sym):
        n = 0
        for symbol, p in by_sym.items():
            if symbol in self.trades or ':' not in (symbol or ''):
                continue
            try:
                entry = float(p.get('entryPrice') or 0)
                qty = float(p.get('contracts') or 0)
                side = p.get('side') or ''
            except Exception:
                continue
            if entry <= 0 or qty <= 0 or side not in ('long', 'short'):
                continue
            sl = float(p.get('stopLossPrice') or 0)
            dist = abs(entry - sl) if sl > 0 else max(entry * 0.0030, 1e-9)
            tp = float(p.get('takeProfitPrice') or 0)
            t = Trade(symbol, side, entry, qty, dist, tp)
            t.be_moved = sl > 0 and ((side == 'long' and sl >= entry) or
                                     (side == 'short' and sl <= entry))
            self.trades[symbol] = t
            log.info('подхвачена позиция %s %s qty=%s @%.6g', symbol, side, qty, entry)
            n += 1
        return n

    async def _move_sl(self, symbol, new_sl):
        try:
            return await self.ex.set_trading_stop(symbol, new_sl, self.trades[symbol].tp)
        except Exception as e:
            log.warning('_move_sl %s: %s', symbol, e)
            return False

    async def _timeout_close(self, symbol, t):
        try:
            await self.ex.close_market(symbol, t.side, t.qty)
        except Exception as e:
            log.warning('тайм-стоп закрытие %s: %s', symbol, e)

    async def _on_closed(self, symbol, t):
        pnl = await self._realized_pnl(symbol, t)
        halted = self.risk.register_close(pnl)
        log.info('ЗАКРЫТО %s: pnl_realized=%.4f, дневной pnl=%.4f%s',
                 symbol, pnl, self.risk.realized_pnl,
                 ' -> СТОП-АУТ' if halted else '')
        self.notify('Закрыто %s: %+.2f$ (день: %+.2f$)' %
                    (symbol, pnl, self.risk.realized_pnl))
        del self.trades[symbol]

    async def _realized_pnl(self, symbol, t):
        trades = await self.ex.my_trades_since(symbol, t.entry_ms)
        pnl = 0.0
        for tr in trades or []:
            info = tr.get('info') or {}
            raw = info.get('closedPnl')
            if raw is not None:
                try:
                    pnl += float(raw)
                except Exception:
                    pass
        if pnl == 0.0:
            pnl = 0.0
        return pnl

    # ---------- стоп-аут ----------

    async def halt_close_all(self):
        if self.halt_executed:
            return
        self.halt_executed = True
        log.warning('Дневной стоп-аут: отмена ордеров + закрытие позиций')
        self.notify('Дневной стоп-аут: всё закрываю')
        for symbol in list(self.trades):
            t = self.trades[symbol]
            try:
                await self.ex.close_market(symbol, t.side, t.qty)
            except Exception as e:
                log.warning('halt close %s: %s', symbol, e)
        for symbol in list(self.trades):
            pos = await self.ex.positions()
            if not any(p.get('symbol') == symbol for p in pos):
                self.risk.register_close(0.0)
                del self.trades[symbol]
        try:
            await self.ex.cancel_all()
        except Exception:
            pass

    def reset_halt(self):
        self.halt_executed = False
