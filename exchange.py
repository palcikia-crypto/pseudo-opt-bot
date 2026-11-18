"""exchange.py — асинхронная обертка Bybit API v5 (ccxt.async_support)."""
import asyncio
import logging
import os

import ccxt.async_support as accxt

from config import BYBIT_API_KEY, BYBIT_SECRET, LEVERAGE, TESTNET

log = logging.getLogger('exchange')

LINEAR_SYMBOLS_TTL = 3600
DEFAULT_HOSTS = 'bybit.com,bytick.com,bybit.nl'
BYBIT_HOSTS = [h.strip() for h in os.getenv('BYBIT_HOSTS', DEFAULT_HOSTS).split(',') if h.strip()]


class Exchange:
    def __init__(self):
        self.ex = self._new_instance(BYBIT_HOSTS[0])
        self._margin_done = set()

    def _new_instance(self, host):
        params = {
            'apiKey': BYBIT_API_KEY,
            'secret': BYBIT_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'swap', 'fetchMarkets': {'types': ['linear']}},
        }
        ex = accxt.bybit(params)
        ex.hostname = host
        if TESTNET:
            ex.set_sandbox_mode(True)
        return ex

    async def load(self):
        last = None
        for host in BYBIT_HOSTS:
            self.ex.hostname = host
            self.ex.markets = None
            self.ex.markets_by_id = None
            try:
                await self.ex.load_markets()
                log.info('load_markets OK (host=%s)', host)
                break
            except Exception as e:
                last = e
                log.warning('load_markets через %s не удался: %s', host, e)
        else:
            raise last
        self._filter_symbols()
        return self.ex.markets

    def _filter_symbols(self):
        bad = [s for s in __import__('config').SYMBOLS if s not in self.ex.markets]
        if bad:
            log.warning('символы не на бирже (исключены): %s', bad)
            for s in bad:
                if s in __import__('config').SYMBOLS:
                    __import__('config').SYMBOLS.remove(s)

    async def close(self):
        await self.ex.close()

    def market(self, symbol):
        return self.ex.market(symbol)

    async def round_amount(self, symbol, qty):
        m = self.market(symbol)
        try:
            q = float(self.ex.amount_to_precision(symbol, qty))
        except Exception:
            return 0.0
        lim = (m.get('limits') or {}).get('amount') or {}
        mn = lim.get('min') or 0.0
        if mn and q < mn:
            return 0.0
        cost = (m.get('limits') or {}).get('cost') or {}
        mn_cost = cost.get('min') or 0.0
        price = None
        return q

    async def price_to_precision(self, symbol, price):
        try:
            return float(self.ex.price_to_precision(symbol, price))
        except Exception:
            return price

    async def candles(self, symbol, timeframe, limit=100):
        return await self.ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    async def order_book(self, symbol, depth=5):
        try:
            return await self.ex.fetch_order_book(symbol, limit=depth)
        except Exception as e:
            log.warning('ob %s: %s', symbol, e)
            return None

    async def free_usdt(self):
        b = await self.ex.fetch_balance()
        usdt = b.get('USDT')
        if isinstance(usdt, dict):
            v = usdt.get('free')
            if v is not None:
                return float(v)
        return float((b.get('free') or {}).get('USDT') or 0.0)

    async def setup_margin(self, symbol):
        if symbol in self._margin_done:
            return
        try:
            await self.ex.set_leverage(LEVERAGE, symbol)
        except Exception:
            pass
        try:
            await self.ex.set_margin_mode('isolated', symbol)
        except Exception:
            pass
        self._margin_done.add(symbol)

    async def place_post_only(self, symbol, side, qty, price, sl, tp):
        params = {'postOnly': True, 'timeInForce': 'IOC'}
        if sl:
            params['stopLoss'] = await self.price_to_precision(symbol, sl)
        if tp:
            params['takeProfit'] = await self.price_to_precision(symbol, tp)
        qty = await self.amount_to_precision(symbol, qty)
        o = await self.ex.create_order(symbol, 'limit', side, qty, None, params)
        return o

    async def amount_to_precision(self, symbol, qty):
        try:
            return float(self.ex.amount_to_precision(symbol, qty))
        except Exception:
            return float(qty)

    async def wait_filled(self, symbol, order_id, timeout_s):
        deadline = asyncio.get_event_loop().time() + timeout_s
        filled = 0.0
        price = 0.0
        while asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.4)
            try:
                o = await self.ex.fetch_order(order_id, symbol)
            except accxt.OrderNotFound:
                return None
            except Exception as e:
                log.debug('fetch_order %s: %s', order_id, e)
                continue
            filled = float(o.get('filled') or 0.0)
            price = float(o.get('average') or o.get('price') or 0.0)
            if o.get('status') in ('closed',) or filled >= 0.999 * float(o.get('amount') or 1):
                return {'filled': filled, 'price': price, 'order': o}
        return None

    async def cancel(self, symbol, order_id):
        try:
            await self.ex.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            log.debug('cancel %s: %s', order_id, e)
            return False

    async def cancel_all(self, symbol=None):
        try:
            await self.ex.cancel_all_orders(symbol)
        except Exception:
            pass

    async def set_trading_stop(self, symbol, sl, tp):
        mid = self.market(symbol)['id']
        fn = getattr(self.ex, 'private_post_v5_position_set_trading_stop', None)
        if fn is None:
            raise RuntimeError('ccxt без поддержки set-trading-stop — обнови ccxt')
        body = {'category': 'linear', 'symbol': mid, 'positionIdx': 0}
        if sl:
            body['stopLoss'] = ('%.10g' % sl)
        if tp:
            body['takeProfit'] = ('%.10g' % tp)
        r = await fn(body)
        ret = (r or {}).get('retCode')
        if ret not in (0, None):
            log.warning('set_trading_stop %s: %s', symbol, r)
            return False
        return True

    async def positions(self):
        try:
            ps = await self.ex.fetch_positions()
        except Exception as e:
            log.warning('positions: %s', e)
            return []
        out = []
        for p in ps:
            try:
                if float(p.get('contracts') or 0) <= 0:
                    continue
            except Exception:
                continue
            if (p.get('symbol') or '').find(':USDT') < 0 and p.get('marginMode') is None:
                continue
            out.append(p)
        return out

    async def open_orders(self, symbol=None):
        try:
            return await self.ex.fetch_open_orders(symbol)
        except Exception:
            return []

    async def close_market(self, symbol, side, qty):
        opp = 'sell' if side == 'long' else 'buy'
        params = {'reduceOnly': True}
        o = await self.ex.create_order(symbol, 'market', opp, qty, None, params)
        return o

    async def my_trades_since(self, symbol, since_ms):
        try:
            return await self.ex.fetch_my_trades(symbol, since=since_ms, limit=50)
        except Exception as e:
            log.debug('my_trades %s: %s', symbol, e)
            return []


class PaperExchange(Exchange):
    """Публичный режим: свечи и стакан без ключей, приватных вызовов нет."""

    def __init__(self):
        params = {'enableRateLimit': True,
                  'options': {'defaultType': 'swap'}}
        self.ex = accxt.bybit(params)
        self._margin_done = set()
