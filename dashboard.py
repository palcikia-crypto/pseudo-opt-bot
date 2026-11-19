"""dashboard.py — веб-панель мониторинга бота.

Читает состояние из paper_state.json (paper-режим) и, если есть, из state.json
(живой RiskManager). Показывает: equity/pnl, открытые позиции с live-ценами
Bybit, историю сделок, win-rate/profit-factor, статус дневного стоп-аута.

Запуск:  python dashboard.py                        (панель сама по себе)
         DASH_PORT=8080 python paper.py             (панель вместе с ботом)
Переменные: DASH_HOST (по умолчанию 127.0.0.1 — наружу не торчит),
DASH_PORT (по умолчанию 8080), PAPER_STATE_FILE.
"""
import asyncio
import datetime as dt
import json
import os

from aiohttp import ClientSession, ClientTimeout, web

import config as C

DASH_HOST = os.getenv('DASH_HOST', '127.0.0.1')
DASH_PORT = int(os.getenv('DASH_PORT', '8080'))
PAPER_STATE_FILE = os.getenv('PAPER_STATE_FILE', 'paper_state.json')
START_EQUITY = float(os.getenv('PAPER_START_EQUITY', '1000'))
TICKER_URL = 'https://api.bybit.com/v5/market/tickers'
PRICE_TTL = 5.0

_session = None
_price_cache = {'ts': 0.0, 'data': {}}


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _compute_stats(trades):
    closed = [t for t in trades if not t.get('open')]
    wins = [t for t in closed if t.get('pnl', 0) > 0]
    losses = [t for t in closed if t.get('pnl', 0) <= 0]
    gp = sum(t['pnl'] for t in wins)
    gl = abs(sum(t['pnl'] for t in losses))
    pf = round(gp / gl, 2) if gl > 0 else (99.0 if gp > 0 else 0.0)
    return {'closed': len(closed), 'wins': len(wins), 'losses': len(losses),
            'winrate': round(100.0 * len(wins) / len(closed), 1) if closed else 0.0,
            'profit_factor': pf or 0.0,
            'best': round(max((t['pnl'] for t in closed), default=0.0), 3),
            'worst': round(min((t['pnl'] for t in closed), default=0.0), 3)}


async def _bybit_prices(symbols):
    now = dt.datetime.now(dt.timezone.utc).timestamp()
    if symbols and now - _price_cache['ts'] > PRICE_TTL:
        global _session
        if _session is None or _session.closed:
            _session = ClientSession(timeout=ClientTimeout(total=4))
        data = {}
        for sym in symbols:
            raw = sym.split('/')[0] + 'USDT'
            try:
                async with _session.get(TICKER_URL,
                                        params={'category': 'linear', 'symbol': raw}) as r:
                    data[sym] = float(((await r.json()).get('result') or {})
                                      .get('list', [{}])[0].get('markPrice') or 0.0)
            except Exception:
                data[sym] = 0.0
        _price_cache.update(ts=now, data=data)
    return _price_cache['data']


def _summary():
    ps = _load_json(PAPER_STATE_FILE) or {}
    rs = _load_json(C.STATE_FILE) or {}
    trades = ps.get('trades', [])
    equity = float(ps.get('equity', START_EQUITY))
    day = ps.get('day')
    day_start = float(ps.get('day_start_equity', equity))
    today = dt.datetime.now(dt.timezone.utc)
    stale = day != today.strftime('%Y-%m-%d')
    day_pnl = round(equity - day_start, 4) if day_start else 0.0
    secs_to_midnight = (dt.datetime.combine(today.date() + dt.timedelta(days=1),
                                            dt.time.min, tzinfo=dt.timezone.utc)
                        - today).total_seconds()
    try:
        mtime = os.path.getmtime(PAPER_STATE_FILE)
    except OSError:
        mtime = 0.0
    return {
        'mode': 'paper',
        'equity': round(equity, 2),
        'pnl_total': round(equity - START_EQUITY, 2),
        'day': day,
        'day_pnl': day_pnl,
        'day_loss_limit_pct': C.DAY_LOSS_PCT * 100.0,
        'day_paused': stale or bool(rs.get('halted', False)),
        'resume_in_s': round(secs_to_midnight) if stale else 0,
        'halted_live': bool(rs.get('halted', False)),
        'live_realized_pnl': rs.get('realized_pnl'),
        'stats': _compute_stats(trades),
        'open_positions': [t for t in trades if t.get('open')],
        'closed_trades': [t for t in trades if not t.get('open')][-30:][::-1],
        'state_age_s': round(today.timestamp() - mtime) if mtime else None,
        'updated_utc': dt.datetime.fromtimestamp(mtime, dt.timezone.utc).isoformat()
                        if mtime else None,
    }


async def api_summary(_request):
    return web.json_response(await _summary_with_marks(_summary()))


async def _summary_with_marks(summary):
    open_pos = summary.get('open_positions') or []
    if not open_pos:
        return summary
    prices = await _bybit_prices(sorted({t['symbol'] for t in open_pos}))
    for t in open_pos:
        mid = prices.get(t['symbol'], 0.0)
        direction = 1 if t['side'] == 'long' else -1
        t['mark'] = mid or None
        t['unrealized'] = round((mid - t['entry']) * t['qty'] * direction, 4) if mid else None
        t['minutes'] = round((dt.datetime.now(dt.timezone.utc).timestamp()
                              - t['entry_ms'] / 1000.0) / 60.0, 1)
    return summary


async def api_health(_request):
    return web.json_response({'ok': True})


def _page():
    return PAGE


async def index(_request):
    return web.Response(text=_page(), content_type='text/html')


def create_app():
    app = web.Application()
    app.router.add_get('/api/summary', api_summary)
    app.router.add_get('/api/health', api_health)
    app.router.add_get('/', index)
    return app


async def start_dashboard_async(host=None, port=None):
    port = port or DASH_PORT
    host = host or DASH_HOST
    runner = web.AppRunner(create_app())
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    return runner, host, port


async def _amain():
    runner, host, port = await start_dashboard_async()
    print(f'[DASH] панель: http://{host}:{port} (PAPER_STATE_FILE={PAPER_STATE_FILE})')
    while True:
        await asyncio.sleep(3600)


PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<title>⚡ Pseudo-Opt Bot — панель</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root { color-scheme: dark; }
body { font-family: system-ui, sans-serif; background:#0d1117; color:#e6edf3; margin:24px; }
h1 { font-size:1.3rem; margin:0 0 4px; } h2 { font-size:1.0rem; color:#8b949e; margin:22px 0 8px; }
.grid { display:flex; gap:18px; flex-wrap:wrap; margin-bottom:10px; }
.card { background:#161b22; border:1px solid #30363d; border-radius:10px; padding:12px 18px; min-width:130px; }
.card .v { font-size:1.5rem; font-weight:700; } .card .l { color:#8b949e; font-size:.78rem; }
.up { color:#3fb950; } .dn { color:#f85149; }
.badge { display:inline-block; padding:3px 10px; border-radius:99px; font-size:.8rem; border:1px solid #30363d; }
.badge.ok { color:#3fb950; } .badge.stop { color:#f85149; }
table { border-collapse:collapse; width:100%; font-size:.86rem; }
th, td { padding:5px 10px; border-bottom:1px solid #21262d; text-align:right; }
th:first-child, td:first-child { text-align:left; }
th { color:#8b949e; font-weight:600; }
.small { color:#8b949e; font-size:.75rem; }
#err { color:#f85149; display:none; margin:8px 0; }
</style></head><body>
<h1>⚡ Pseudo-Opt Bot <span id="badge" class="badge">…</span></h1>
<div class="small" id="meta">—</div>
<div id="err"></div>
<h2>Состояние</h2>
<div class="grid">
  <div class="card"><div class="v" id="equity">—</div><div class="l">Equity, USDT</div></div>
  <div class="card"><div class="v" id="pnl">—</div><div class="l">PnL всего</div></div>
  <div class="card"><div class="v" id="daypnl">—</div><div class="l">PnL сегодня (стоп −<span id="dlp">4</span>%)</div></div>
  <div class="card"><div class="v" id="wr">—</div><div class="l">Win-rate</div></div>
  <div class="card"><div class="v" id="pf">—</div><div class="l">Profit factor</div></div>
  <div class="card"><div class="v" id="closed">—</div><div class="l">Сделок закрыто</div></div>
</div>
<h2>Открытые позиции</h2>
<table><thead><tr>
<th>Символ</th><th>Side</th><th>Вход</th><th>Mark</th><th>SL</th><th>TP</th><th>BE</th>
<th>Unrealized</th><th>Минут</th></tr></thead><tbody id="open"></tbody></table>
<h2>Закрытые сделки (последние 30)</h2>
<table><thead><tr>
<th>Символ</th><th>Side</th><th>Вход</th><th>Выход</th><th>PnL</th><th>Причина</th>
<th>Длительность, мин</th></tr></thead><tbody id="hist"></tbody></table>
<p class="small">данные: paper_state.json + публичный Bybit v5 • автообновление каждые 3 с</p>
<script>
const fmt = (x, d=6) => (x == null ? '—' : (+x).toFixed(d));
async function tick() {
  try {
    const s = await (await fetch('/api/summary')).json();
    document.getElementById('err').style.display='none';
    const eq = document.getElementById('equity');
    eq.textContent = s.equity; eq.className = 'v ' + (s.pnl_total>=0?'up':'dn');
    const p = document.getElementById('pnl');
    p.textContent = (s.pnl_total>=0?'+':'') + s.pnl_total;
    p.className = 'v ' + (s.pnl_total>=0?'up':'dn');
    const dp = document.getElementById('daypnl');
    dp.textContent = (s.day_pnl>=0?'+':'') + s.day_pnl;
    dp.className = 'v ' + (s.day_pnl>=0?'up':'dn');
    document.getElementById('dlp').textContent = s.day_loss_limit_pct;
    document.getElementById('wr').textContent = s.stats.winrate + '%';
    document.getElementById('pf').textContent = s.stats.profit_factor;
    document.getElementById('closed').textContent = s.stats.closed;
    const b = document.getElementById('badge');
    if (s.halted_live) { b.textContent='СТОП-АУТ'; b.className='badge stop'; }
    else if (s.day_paused) { b.textContent='ПАУЗА до 00:00 UTC (' + s.resume_in_s + ' c)'; b.className='badge stop'; }
    else if (s.state_age_s !== null && s.state_age_s < 120) { b.textContent='ЖИВОЙ'; b.className='badge ok'; }
    else { b.textContent='нет обновлений'; b.className='badge stop'; }
    document.getElementById('meta').textContent =
      'день: ' + s.day + ' • закрыто: ' + s.stats.closed + ' (W ' + s.stats.wins +
      '/L ' + s.stats.losses + ') • best/worst: ' + s.stats.best + '/' + s.stats.worst +
      ' • state: ' + (s.updated_utc||'—');
    document.querySelector('#open').innerHTML = (s.open_positions||[]).map(t => {
      const u = t.unrealized == null ? '—' : (t.unrealized>=0?'+':'') + t.unrealized;
      return `<tr><td>${t.symbol}</td><td>${t.side}</td><td>${t.entry}</td>
        <td>${fmt(t.mark)}</td><td class="dn">${fmt(t.sl)}</td><td class="up">${fmt(t.tp)}</td>
        <td>${t.be?'✓':''}</td><td class="${t.unrealized>=0?'up':'dn'}">${u}</td><td>${t.minutes}</td></tr>`;
    }).join('') || '<tr><td colspan="9" class="small">нет открытых позиций</td></tr>';
    document.querySelector('#hist').innerHTML = (s.closed_trades||[]).map(t => {
      const mins = Math.round((t.exit_ms - t.entry_ms)/60000);
      return `<tr><td>${t.symbol}</td><td>${t.side}</td><td>${t.entry}</td><td>${t.exit}</td>
        <td class="${t.pnl>=0?'up':'dn'}">${(t.pnl>=0?'+':'')}${t.pnl}</td>
        <td>${t.reason}</td><td>${mins}</td></tr>`;
    }).join('') || '<tr><td colspan="7" class="small">пока нет закрытых сделок</td></tr>';
  } catch (e) {
    const el = document.getElementById('err');
    el.style.display='block'; el.textContent = 'Ошибка связи: ' + e;
  }
}
tick(); setInterval(tick, 3000);
</script></body></html>"""


if __name__ == '__main__':
    asyncio.run(_amain())
