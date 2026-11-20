"""research/evo.py — эволюционный движок и демон (раунд 9).

Что перебирает (мутации + кроссовер + отбор):
  - монеты (15), ТФ (15м/60м + 1м для микро-идей), семейства сигналов
    (9 lab-идей + 6 micro2 + union-комбинации),
  - параметры масок (atr_min_pct, vol_mult) и ВЫХОДОВ (sl_atr_mult, rr,
    trail_atr, sl_max_pct, max_hold, time_stop, be_trigger),
  - отбор по fitness = sum_R × доля месяцев в + (train-окно БЕЗ последних 14д),
  - валидация top-особей на свежих 14д (val) + option_edge (годен ли сигнал под
    настоящий опцион), slippage_report (loblog), shadow_report (live 1-в-1).

Daemon (循环 30 мин): инкремент данных (refresh_tail), поколение, отчёт в
research/results/evo.json + research/results/panel_meta.json['evo'].

Запуск: ./venv/bin/python research/evo.py --generations 40 --pop 40
        ./venv/bin/python research/evo.py --daemon
"""
import argparse
import copy
import datetime
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import research.btcore as bt
import research.deep as deep

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

LAB_IDEAS = ('trend_bb15', 'trend_bb1h', 'squeeze', 'mr_rsi5', 'mr_vwap',
             'donchian', 'impulse', 'pullback', 'btc_led')
MICRO_IDEAS = ('micro2:climax_fade', 'micro2:nr_break', 'micro2:vwap_z',
               'micro2:vol_shock', 'micro2:wick_chain2', 'micro2:doji_exhaust')
EXIT_NAMES = {k: deep.DEEP_EXITS[k]
              for k in ('m1', 'm1_trail', 'wide:3.5x3.0', 'tf_trail', 'tf_wide')}
EXTRA_EXITS = {'fade20': dict(sl_atr_mult=1.2, exit='time', max_hold_min=20,
                             sl_max_pct=0.015)}
EXITS = dict(EXIT_NAMES, **EXTRA_EXITS)
MAX_TF = 4  # 1, 15, 60

SEEDS = [
    ('APTUSDT', 15, 'mr_rsi5', 'tf_trail'),
    ('XRPUSDT', 60, 'mr_vwap', 'm1_trail'),
    ('SOLUSDT', 60, 'mr_vwap', 'm1_trail'),
    ('BTCUSDT', 15, 'union:impulse+pullback', 'tf_wide'),
    ('SEIUSDT', 15, 'micro2:doji_exhaust', 'm1_trail'),
]
BASE_PARAMS = dict(atr_min_pct=0.03, vol_mult=1.3, exit=dict(EXITS['m1_trail']))


def base_exit(name):
    return copy.deepcopy(EXITS[name])


def rand_params(rng):
    return dict(atr_min_pct=round(rng.uniform(0.02, 0.08), 3),
                vol_mult=round(rng.uniform(1.0, 2.0), 2),
                exit_name=rng.choice(list(EXITS)),
                exit=dict(rng.choice(list(EXITS.values()))))


def fitness_from(st):
    return st['fitness'] if st['n'] >= 10 else -999.0


class Evo:
    def __init__(self, pop_size=40, seed=7):
        self.rng = random.Random(seed)
        self.pop_size = pop_size
        self.hall = []
        self.gen = 0

    # --- операции ---
    def mutate(self, g):
        c = g.clone()
        r = self.rng
        p = dict(c['params'])
        if r.random() < 0.6:
            p['atr_min_pct'] = round(max(0.01, min(0.12,
                float(p['atr_min_pct']) * r.uniform(0.75, 1.3))), 3)
        if r.random() < 0.6:
            p['vol_mult'] = round(max(0.8, min(2.5,
                float(p['vol_mult']) * r.uniform(0.7, 1.4))), 2)
        ex = dict(p['exit'])
        for k, rng_ in (('sl_atr_mult', (0.02, 5.0)), ('rr', (0.01, 6.0)),
                        ('trail_atr', (0.01, 4.0))):
            if k in ex and r.random() < 0.7:
                ex[k] = round(max(rng_[0], min(rng_[1],
                              float(ex[k]) * r.uniform(0.8, 1.25))), 2)
        for k, rng_ in (('sl_max_pct', (0.006, 0.05)), ('max_hold_min', (10, 240)),
                        ('time_stop_min', (120, 1440))):
            if k in ex and r.random() < 0.5:
                ex[k] = round(r.uniform(*rng_), 3 if k == 'sl_max_pct' else 0)
        if 'be_trigger' in ex and r.random() < 0.4:
            ex['be_trigger'] = round(max(0.8, min(2.5, ex['be_trigger'] * r.uniform(0.8, 1.25))), 2)
        p['exit'] = ex
        c['params'] = p
        if r.random() < 0.25:
            c['symbol'] = r.choice(bt.full)
        if r.random() < 0.25:
            c['idea'] = r.choice(LAB_IDEAS + MICRO_IDEAS)
        if c['idea'] == 'btc_led' and str(c['symbol']).startswith('BTC'):
            c['idea'] = 'impulse'
        if r.random() < 0.2:
            c['exit_name'] = r.choice(list(EXITS))
            c['params']['exit'] = base_exit(c['exit_name'])
        return c

    def crossover(self, a, b):
        c = a.clone()
        c['symbol'] = self.rng.choice([a['symbol'], b['symbol']])
        c['tf'] = self.rng.choice([a['tf'], b['tf']])
        c['idea'] = self.rng.choice([a['idea'], b['idea']])
        p = dict(a['params'])
        p2 = b['params']
        p['atr_min_pct'] = self.rng.choice([p['atr_min_pct'], p2['atr_min_pct']])
        p['vol_mult'] = self.rng.choice([p['vol_mult'], p2['vol_mult']])
        p['exit'] = dict(self.rng.choice([a['params']['exit'], b['params']['exit']]))
        c['params'] = p
        return c

    def seed_population(self):
        pop = []
        for sym, tf, idea, ex in SEEDS:
            g = bt.Genome.new(sym, tf, idea, ex,
                              dict(atr_min_pct=0.03, vol_mult=1.3,
                                   exit=base_exit(ex)))
            pop.append(g)
        while len(pop) < self.pop_size:
            g = bt.Genome.new(self.rng.choice(bt.full),
                              15 if self.rng.random() < 0.6 else 60,
                              self.rng.choice(LAB_IDEAS + MICRO_IDEAS),
                              self.rng.choice(list(EXITS)), rand_params(self.rng))
            if g['idea'] == 'btc_led' and str(g['symbol']).startswith('BTC'):
                g['idea'] = 'impulse'
            pop.append(self.mutate(g))
        self.pop = pop

    def run(self, views, t0, t_train_end, t_end, generations, log=print):
        evals = 0
        for gen in range(generations):
            scored = []
            for g in self.pop:
                st, _ = bt.evaluate(views, g, t0, t_train_end)
                evals += 1
                scored.append((fitness_from(st), g, st))
            scored.sort(key=lambda x: x[0], reverse=True)
            self.gen += 1
            elite = scored[:max(2, self.pop_size // 6)]
            best_fit, best_g, best_st = scored[0]
            if not self.hall or best_fit > self.hall[0][0]:
                self.hall.insert(0, (best_fit, best_g, dict(best_st), self.gen))
                self.hall = self.hall[:8]
                log('[gen %d] NEW BEST %.2fR fit | %s | %s' % (
                    self.gen, best_fit, best_g.key, best_st), flush=True)
            nxt = [g for _, g, _, _ in [(h[0], h[1], h[2], h[3]) for h in self.hall[:3]]]
            nxt += [g for _, g, _ in elite]
            while len(nxt) < self.pop_size:
                a = self.tourn(scored)
                c = self.tourn(scored)
                child = self.crossover(a, c) if self.rng.random() < 0.5 else a.clone()
                if self.rng.random() < 0.85:
                    child = self.mutate(child)
                nxt.append(child)
            self.pop = nxt
        return evals

    def tourn(self, scored, k=3):
        cands = self.rng.sample(scored, min(k, len(scored)))
        return max(cands, key=lambda x: x[0])[1]


def snapshot_report(evo, views, t_val_start, t_end):
    rep = dict(measured=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               gen=evo.gen, pop=len(evo.pop), hall=[])
    for fit, g, st, gen_num in evo.hall[:6]:
        stv, tr_val = bt.evaluate(views, g, t_val_start, t_end)
        oe = bt.option_edge(tr_val, hold_min_med=max(g['tf'], 15) * 3)
        rep['hall'].append(dict(fitness=round(fit, 2), gen=gen_num, genome=g.to_json(),
                                train=dict(n=st['n'], sum_R=st['sum_R'],
                                           months_pos=st.get('months_pos')),
                                val=dict(n=stv['n'], sum_R=stv['sum_R'],
                                         wr=stv['win_rate'],
                                         months_pos=stv.get('months_pos')),
                                option=oe))
    rep['slippage'] = bt.slippage_report()
    rep['shadow'] = bt.shadow_report()
    return rep


def save_report(rep):
    os.makedirs(RES, exist_ok=True)
    json.dump(rep, open(os.path.join(RES, 'evo.json'), 'w'), indent=1,
              ensure_ascii=False)
    pm = os.path.join(RES, 'panel_meta.json')
    if os.path.exists(pm):
        meta = json.load(open(pm))
        h = rep['hall'][0] if rep['hall'] else None
        meta['evo'] = dict(
            gen=rep['gen'], pop=rep['pop'],
            best=(h['genome']['symbol'] + ' ' + h['genome']['idea'] + ' ' +
                  h['genome']['exit_name'] + ' | train %+.1fR/n=%d | val %+.1fR/n=%d | %s'
                  % (h['train']['sum_R'], h['train']['n'],
                     h['val']['sum_R'], h['val']['n'],
                     'option-viable' if h['option'] and h['option']['edge_bp'] > 0
                     else 'opt-no')) if h else 'нет',
            shadow=rep['shadow'],
            slippage={k: v.get('median_spread_bps') for k, v in
                      rep['slippage'].items() if isinstance(v, dict)})
        json.dump(meta, open(pm, 'w'), indent=1, ensure_ascii=False)


def load_state(path):
    if os.path.exists(path):
        d = json.load(open(path))
        evo = Evo()
        evo.gen = d.get('gen', 0)
        evo.hall = [(h['fitness'], bt.Genome.from_json(h['genome']), h['train'],
                     h['gen']) for h in d.get('hall', [])]
        return evo
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--generations', type=int, default=30)
    ap.add_argument('--pop', type=int, default=40)
    ap.add_argument('--daemon', action='store_true')
    ap.add_argument('--cycle-sec', type=int, default=1800)
    args = ap.parse_args()

    state_path = os.path.join(RES, 'evo_state.json')
    evo = load_state(state_path) or Evo(pop_size=args.pop)
    evo.pop_size = args.pop

    full = None
    while True:
        if full is None or args.daemon:
            for sym in (full or []):
                try:
                    bt.refresh_tail(sym, tail_days=3)
                except Exception as e:
                    print('[refresh %s] fail: %s' % (sym, str(e)[:80]), flush=True)
            t_start_data = time.time()
            print('[data] загрузка 190д...', flush=True)
            bt.full, raw, views = bt.load_views()
            t0 = int(min(l.ts[0] for l in views[15].values()))
            t1 = int(max(l.ts[-1] for l in views[15].values()))
            t_train_end = t1 - 14 * bt.DAY
            t_val_start = t_train_end
            print('[data] ok %.0fs' % (time.time() - t_start_data), flush=True)
        if not hasattr(evo, 'pop') or not getattr(evo, 'pop', None):
            evo.seed_population()
        evals = evo.run(views, t0, t_train_end, t1, args.generations)
        rep = snapshot_report(evo, views, t_val_start, t1)
        rep['evals'] = evals
        save_report(rep)
        json.dump(dict(gen=evo.gen, hall=[dict(fitness=h[0], genome=h[1].to_json(),
                                               train=h[2], gen=h[3])
                                         for h in evo.hall]),
                  open(state_path, 'w'), indent=1, ensure_ascii=False)
        if rep['hall']:
            hg = rep['hall'][0]['genome']
            print('[evo] gen=%d evals=%d best=%s' % (
                evo.gen, evals,
                '%s|%dm|%s|%s' % (hg['symbol'], hg['tf'], hg['idea'],
                                  hg['exit_name'])), flush=True)
        if not args.daemon:
            break
        time.sleep(args.cycle_sec)


if __name__ == '__main__':
    main()
