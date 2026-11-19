"""snapshot.py — готовит компактный state.json для веб-панели из paper_state.json.

publish(): через git plumbing (без изменения рабочей копии) собирает
orphan-ветку PANEL_BRANCH из panel/index.html + state.json + .nojekyll и пушит её.

Запуск там, где работает бот:   PANEL_BRANCH=panel-site python panel/snapshot.py
Зациклить (VPS/песочница):      PANEL_SYNC_SEC=600 sh panel/sync.sh
"""
import datetime as dt
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C

PAPER_STATE_FILE = os.getenv('PAPER_STATE_FILE', 'paper_state.json')
PANEL_BRANCH = os.getenv('PANEL_BRANCH', 'panel-site')
START_EQUITY = float(os.getenv('PAPER_START_EQUITY', '1000'))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _raw_symbols():
    labels, raw = [], []
    for sym in C.SYMBOLS:
        base = sym.split('/')[0]
        labels.append(base)
        raw.append(base + 'USDT')
    return {'raw': raw, 'labels': labels}


def snapshot():
    try:
        with open(PAPER_STATE_FILE) as f:
            ps = json.load(f)
    except Exception:
        ps = {}
    trades = ps.get('trades', [])
    equity = float(ps.get('equity', START_EQUITY))
    today = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d')
    stale = ps.get('day') != today
    try:
        mtime = os.path.getmtime(PAPER_STATE_FILE)
    except OSError:
        mtime = 0.0
    halted_live = False
    try:
        with open(C.STATE_FILE) as f:
            halted_live = bool(json.load(f).get('halted', False))
    except Exception:
        pass
    return {
        'mode': 'paper',
        'equity': round(equity, 2),
        'start_equity': START_EQUITY,
        'pnl_total': round(equity - START_EQUITY, 2),
        'day': ps.get('day'),
        'day_start_equity': ps.get('day_start_equity'),
        'day_paused': stale,
        'halted': halted_live,
        'day_loss_limit_pct': C.DAY_LOSS_PCT * 100.0,
        'max_positions': C.MAX_POSITIONS,
        'risk_per_trade_pct': C.RISK_PER_TRADE_PCT * 100.0,
        'updated_utc': (dt.datetime.fromtimestamp(mtime, dt.timezone.utc).isoformat()
                        if mtime else None),
        'published_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'age_minutes': ((dt.datetime.now(dt.timezone.utc).timestamp() - mtime) / 60.0)
                        if mtime else None,
        'symbols_raw': _raw_symbols(),
        'trades': trades,
    }


def _git(*args, stdin=None):
    res = subprocess.run(['git', '-C', REPO, *args], input=stdin,
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError('git %s failed: %s' % (args[0], res.stderr.strip()))
    return res.stdout.strip()


def publish(snap):
    snap_json = json.dumps(snap, indent=1)

    def blob(text):
        return _git('hash-object', '-w', '-t', 'blob', '--stdin', stdin=text)

    b_index = blob(open(os.path.join(REPO, 'panel', 'index.html')).read())
    b_state = blob(snap_json)
    b_nojek = blob('')

    entries = ('100644 blob %s\tindex.html\n'
               '100644 blob %s\tstate.json\n'
               '100644 blob %s\t.nojekyll\n') % (b_index, b_state, b_nojek)
    tree = _git('mktree', stdin=entries)

    ts = dt.datetime.now(dt.timezone.utc).strftime('%FT%TZ')
    commit = _git('commit-tree', tree, '-m', 'panel: snapshot ' + ts)
    _git('update-ref', 'refs/heads/' + PANEL_BRANCH, commit)
    subprocess.check_call(['git', '-C', REPO, 'push', 'origin',
                           'refs/heads/' + PANEL_BRANCH + ':refs/heads/' + PANEL_BRANCH,
                           '--force'])
    print('[SNAPSHOT] опубликовано: %s (%s)' % (PANEL_BRANCH, ts))
    return ts


if __name__ == '__main__':
    publish(snapshot())
