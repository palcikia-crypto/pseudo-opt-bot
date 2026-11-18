"""paper.py — live paper-trading на публичных данных Bybit, ключи не нужны.

Читает реальные 1m/5m свечи и стакан, гоняет ту же стратегию
(indicators.py + strategy.py), симулирует сделки с реальными комиссиями
(maker 0.02% на вход/TP, taker 0.055% на SL и market-выходы),
безубыток и дневной стоп — как в бою. Состояние: paper_state.json.
Запуск: python paper.py (вечно) | PAPER_MAX_RUNTIME=300 python paper.py (слайс — для cron)
"""
import asyncio
import datetime as dt
import json
import os
import time

import config as C
import indicators as I
from strategy import check_signal

PAPER_STATE_FILE = os.getenv("PAPER_STATE_FILE", "paper_state.json")
POLL_SEC = float(os.getenv("PAPER_POLL_SEC", "5"))
PAPER_MAX_RUNTIME = float(os.getenv("PAPER_MAX_RUNTIME", "0"))
MAKER_FEE = 0.0002
TAKER_FEE = 0.00055
TIME_STOP_MIN = C.TIME_STOP_MIN
BE_TRIGGER = C.BE_TRIGGER
BE_OFFSET_PCT = C.BE_OFFSET_PCT
DAY_LOSS_PCT = C.DAY_LOSS_PCT * 100.0
START_EQUITY = float(os.getenv("PAPER_START_EQUITY", "1000"))


def atr_pct(atr, close):
    return atr / close if close else 0.0


class PaperBot:
    def __init__(self, ex):
        self.ex = ex
        self.state = self._load()
        self.equity = float(self.state.get("equity", START_EQUITY))
        self.day = self.state.get("day")
        self.day_start_eq = float(self.state.get("day_start_equity", self.equity))
        self.trades = self.state.get("trades", [])
        self.open = {}
        self.day_paused = None
        self._adopt_open()

    def _load(self):
        if os.path.exists(PAPER_STATE_FILE):
            with open(PAPER_STATE_FILE) as f:
                return json.load(f)
        return {"equity": START_EQUITY, "trades": []}

    def _save(self):
        self.state.update({"equity": self.equity, "day": self.day,
                           "day_start_equity": self.day_start_eq,
                           "trades": self.trades[-400:]})
        with open(PAPER_STATE_FILE, "w") as f:
            json.dump(self.state, f, indent=1)

    def _adopt_open(self):
        for t in self.trades:
            if t.get("open"):
                self.open[t["symbol"]] = t

    def _roll_day(self):
        today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        if self.day != today:
            self.day = today
            self.day_start_eq = self.equity
            self.day_paused = None
            self._save()

    def _day_loss_hit(self):
        if self.day_start_eq <= 0:
            return False
        loss = (self.equity - self.day_start_eq) / self.day_start_eq * 100.0
        return loss <= -DAY_LOSS_PCT

    def _day_pause_ok(self):
        return not getattr(self, "day_paused", None) == self.day

    async def scan(self):
        """Одна итерация: сигналы + сопровождение открытых сделок."""
        self._roll_day()
        if not self._day_pause_ok():
            return {}
        sigs = {}
        for sym in C.SYMBOLS:
            df5 = await self.ex.candles(sym, "5m", 100)
            df1 = await self.ex.candles(sym, "1m", 100)
            if df5 is None or df1 is None or len(df1) < 60:
                continue
            sig = check_signal(df5, df1, None)
            if sig:
                sigs[sym] = sig
        ob_cache = {}
        for sym in list(sigs):
            ob = await self.ex.order_book(sym, 5)
            ob_cache[sym] = ob
            if not ob:
                continue
            side, atr1, close = sigs[sym]
            imb = I.ob_imbalance(ob["bids"], ob["asks"], C.OB_DEPTH)
            ok = (imb is not None and imb >= C.OB_IMB) if side == "long" else \
                 (imb is not None and imb <= 1.0 / C.OB_IMB)
            if not ok:
                continue
            self._maybe_open(sym, side, atr1, close, ob)
        await self._monitor(ob_cache)
        if self._day_loss_hit():
            self.day_paused = self.day
            await self._flatten("day_stop")
        return sigs

    @staticmethod
    def _sig_note_exit(t, mid, reason):
        pnl = (mid - t["entry"]) * t["qty"] * (1 if t["side"] == "long" else -1)
        return pnl > 0

    def _maybe_open(self, sym, side, atr1, close, ob):
        if sym in self.open or not ob or len(self.open) >= C.MAX_POSITIONS:
            return
        import risk_manager as RM
        sl_d = RM.sl_distance(close, atr1)
        if sl_d <= 0:
            return
        tp_d = RM.tp_distance(sl_d)
        risk_usdt = self.equity * C.RISK_PER_TRADE_PCT
        qty = risk_usdt / sl_d
        if qty <= 0:
            return
        best = ob["bids"][0][0] if side == "long" else ob["asks"][0][0]
        entry = float(best)
        d = sl_d if side == "long" else -sl_d
        t = {
            "symbol": sym, "side": side, "qty": qty, "entry": entry,
            "entry_ms": int(time.time() * 1000), "open": True, "be": False,
            "sl": entry - d, "tp": entry + (tp_d if side == "long" else -tp_d),
            "sl0": entry - d, "tp0": entry + (tp_d if side == "long" else -tp_d),
            "fees": round(entry * qty * MAKER_FEE, 6),
        }
        self.open[sym] = t
        self.trades.append(t)
        self._save()

    async def _monitor(self, ob_cache):
        for sym, t in list(self.open.items()):
            ob = ob_cache.get(sym)
            if not ob:
                ob = await self.ex.order_book(sym, 5)
            if not ob:
                continue
            mid = (float(ob["bids"][0][0]) + float(ob["asks"][0][0])) / 2
            long = t["side"] == "long"
            profit_dist = (mid - t["entry"]) if long else (t["entry"] - mid)
            sl_d = abs(t["sl0"] - t["entry"])
            if not t["be"] and profit_dist >= BE_TRIGGER * sl_d:
                t["be"] = True
                t["sl"] = t["entry"] + (BE_OFFSET_PCT * t["entry"] if long
                                        else -BE_OFFSET_PCT * t["entry"])
                self._save()
            hit_sl = mid <= t["sl"] if long else mid >= t["sl"]
            hit_tp = mid >= t["tp"] if long else mid <= t["tp"]
            mins = (time.time() * 1000 - t["entry_ms"]) / 60000.0
            if hit_sl:
                await self._close(t, t["sl"], "sl" if not t["be"] else "breakeven", TAKER_FEE)
            elif hit_tp:
                await self._close(t, t["tp"], "tp")
            elif mins > TIME_STOP_MIN:
                await self._close(t, mid, "time_stop", TAKER_FEE)

    async def _close(self, t, px, reason, fee_rate=MAKER_FEE):
        exit_px = float(px)
        fee = exit_px * t["qty"] * fee_rate
        pnl = (exit_px - t["entry"]) * t["qty"] * (1 if t["side"] == "long" else -1)
        pnl -= t["fees"] + fee
        self.equity += pnl
        t.update({"open": False, "exit": exit_px, "pnl": round(pnl, 4),
                  "reason": reason, "exit_ms": int(time.time() * 1000)})
        self.open.pop(t["symbol"], None)
        self._save()
        tag = "+" if pnl >= 0 else ""
        print(f"[CLOSE] {t['symbol']} {t['side']} {reason}: {tag}{pnl:.4f} USDT "
              f"(eq={self.equity:.2f})")

    async def _flatten(self, reason):
        for sym, t in list(self.open.items()):
            ob = await self.ex.order_book(sym, 5)
            if not ob:
                continue
            px = float(ob["asks"][0][0]) if t["side"] == "long" else float(ob["bids"][0][0])
            await self._close(t, px, reason, TAKER_FEE)
        print(f"[DAY-STOP] плоско, пауза до следующего UTC-дня (day={self.day})")

    def stats(self):
        closed = [t for t in self.trades if not t.get("open")]
        wins = [t for t in closed if t["pnl"] > 0]
        losses = [t for t in closed if t["pnl"] <= 0]
        gp = sum(t["pnl"] for t in wins)
        gl = abs(sum(t["pnl"] for t in losses))
        pf = round(gp / gl, 2) if gl > 0 else (99.0 if gp > 0 else 0.0)
        return {"trades": len(closed), "open": len(self.open),
                "winrate": round(100.0 * len(wins) / len(closed), 1) if closed else 0.0,
                "profit_factor": pf, "pnl": round(self.equity - START_EQUITY, 4),
                "equity": round(self.equity, 2)}


async def _load_with_retry(ex, attempts=3):
    for i in range(1, attempts + 1):
        try:
            await ex.load()
            return
        except Exception as e:
            print(f"[LOAD] попытка {i}/{attempts} не удалась: {type(e).__name__}: {e}")
            await asyncio.sleep(5.0 * i)
    raise RuntimeError("bybit load_markets не удался на всех хостах/попытках")


async def run_forever():
    from exchange import PaperExchange
    ex = PaperExchange()
    try:
        await _load_with_retry(ex)
        bot = PaperBot(ex)
        await _run_loop(ex, bot)
    finally:
        try:
            await ex.close()
        except Exception:
            pass


async def _run_loop(ex, bot):
    s = bot.stats()
    print(f"[PAPER] старт: equity={s['equity']} closed={s['trades']} "
          f"WR={s['winrate']}% PF={s['profit_factor']}")
    t_start = time.time()
    try:
        while True:
            try:
                if PAPER_MAX_RUNTIME > 0 and time.time() - t_start > PAPER_MAX_RUNTIME:
                    print("[PAPER] слайс завершён, состояние сохранено")
                    break
                sigs = await bot.scan()
                if sigs:
                    print("[SIG]", ", ".join(f"{k} {v[0]} (ATR%={100*atr_pct(v[1], v[2]):.3f})"
                                             for k, v in sigs.items()))
                s = bot.stats()
                print(f"[PAPER] {dt.datetime.now(dt.timezone.utc):%H:%M:%S} "
                      f"eq={s['equity']} open={s['open']} closed={s['trades']} "
                      f"WR={s['winrate']}% PF={s['profit_factor']} pnl={s['pnl']:+.2f}")
            except Exception as e:
                print("[ERR]", type(e).__name__, e)
            await asyncio.sleep(max(POLL_SEC, 2.0))
    finally:
        bot._save()
        s = bot.stats()
        print(f"[FINAL] equity={s['equity']} closed={s['trades']} WR={s['winrate']}% "
              f"PF={s['profit_factor']} pnl={s['pnl']:+.2f} open={s['open']}")


if __name__ == "__main__":
    asyncio.run(run_forever())
