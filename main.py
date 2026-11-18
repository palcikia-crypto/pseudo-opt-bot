"""main.py — главный асинхронный цикл: 15 пар, сигналы на закрытии 1m, мониторинг каждые 2с."""
import asyncio
import datetime
import logging
import signal
import sys
import time

import ccxt.async_support as accxt
import aiohttp

import config as C
from exchange import Exchange
from risk_manager import RiskManager
from strategy import check_signal
from trader import Trader

log = logging.getLogger('main')


def setup_logging():
    fmt = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    logging.basicConfig(
        level=getattr(logging, C.LOG_LEVEL.upper(), logging.INFO),
        format=fmt,
        handlers=[logging.FileHandler(C.LOG_FILE), logging.StreamHandler(sys.stdout)],
    )


async def tg_notify(msg):
    if not C.TG_TOKEN or not C.TG_CHAT_ID:
        return
    try:
        url = 'https://api.telegram.org/bot%s/sendMessage' % C.TG_TOKEN
        async with aiohttp.ClientSession() as s:
            await s.post(url, json={'chat_id': C.TG_CHAT_ID, 'text': msg})
    except Exception:
        pass


async def fetch_all_candles(ex, symbols):
    async def one(sym):
        try:
            c1 = await ex.candles(sym, C.TF_ENTRY, C.CANDLE_LIMIT)
            c5 = await ex.candles(sym, C.TF_TREND, C.CANDLE_LIMIT)
            return sym, (c5, c1)
        except Exception as e:
            log.warning('candles %s: %s', sym, e)
            return sym, None
    out = {}
    for sym, r in await asyncio.gather(*(one(s) for s in symbols)):
        out[sym] = r
    return out


async def entry_loop(ex, risk, trader, symbols):
    while True:
        try:
            await sleep_to_next_minute(2.0)
            balance = await ex.free_usdt()
            risk.rollover(balance)
            if risk.halted:
                await trader.halt_close_all()
                await asyncio.sleep(30)
                continue
            candles = await fetch_all_candles(ex, symbols)
            signals = {}
            for sym, cc in candles.items():
                if not cc or trader is None or sym in trader.trades:
                    continue
                sig = check_signal(cc[0], cc[1], None)
                if sig:
                    signals[sym] = sig
            for sym, (side, atr1, close) in signals.items():
                ob = await ex.order_book(sym, C.OB_DEPTH)
                if ob:
                    import indicators as _I
                    imb = _I.ob_imbalance(ob.get('bids') or [], ob.get('asks') or [], C.OB_DEPTH)
                    ok = (imb is not None and imb >= C.OB_IMB) if side == 'long' else \
                         (imb is not None and (1.0 / imb) >= C.OB_IMB)
                    if not ok:
                        continue
                else:
                    continue
                try:
                    await trader.try_enter(sym, side, atr1, close, ob)
                except Exception as e:
                    log.warning('try_enter %s: %s', sym, e)
        except accxt.RateLimitExceeded as e:
            log.warning('rate limit: %s', e)
            await asyncio.sleep(30)
        except Exception:
            log.exception('entry loop error')
            await asyncio.sleep(10)


async def monitor_loop(ex, risk, trader):
    while True:
        try:
            await asyncio.sleep(C.POLL_SEC)
            await trader.monitor_once()
        except accxt.RateLimitExceeded:
            await asyncio.sleep(30)
        except Exception:
            log.exception('monitor error')
            await asyncio.sleep(5)


async def sleep_to_next_minute(offset=2.0):
    now = time.time()
    target = (int(now // 60) + 1) * 60 + offset
    await asyncio.sleep(max(0.0, target - now))


async def amain():
    setup_logging()
    log.info('=== псевдо-опционный скальпер Launch | testnet=%s | символов %d ===',
             C.TESTNET, len(C.SYMBOLS))
    if not C.BYBIT_API_KEY:
        log.error('нет BYBIT_API_KEY в .env — останавливаюсь')
        return
    ex = Exchange()
    await ex.load()
    risk = RiskManager()
    trader = Trader(ex, risk, notify=tg_notify)

    for sym in C.SYMBOLS:
        await ex.setup_margin(sym)

    await trader._adopt({p.get('symbol'): p for p in await ex.positions()})
    log.info('подхвачено позиций при старте: %d', len(trader.trades))
    await tg_notify('Бот запущен (testnet=%s), позиций: %d' %
                    (C.TESTNET, len(trader.trades)))

    stop = asyncio.Event()
    loop = asyncio.get_event_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(s, stop.set)
        except NotImplementedError:
            pass

    tasks = [asyncio.create_task(entry_loop(ex, risk, trader, C.SYMBOLS)),
             asyncio.create_task(monitor_loop(ex, risk, trader))]
    await stop.wait()
    log.info('остановка...')
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await ex.close()


if __name__ == '__main__':
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        pass
