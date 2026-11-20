"""research/optevo.py — эволюция под ОПЦИОННЫЙ выйгрыш (раунд 9b).

Перебирает: монеты (8 ликвидных) × TF {1,15,60} × 15 идей × экспирации
{20,30,60,120,240,480} мин × prem_k {0.7,0.85,1.0} × параметры масок.
Против perp-эволюции (evo.py) тут другой фитнес — ванильная выплата
опциона (bp), цифровой контраст как второй критерий. Символы урезаны:
демон живёт рядом с evo.py в одной песочнице (RAM).

Запуск:  ./venv/bin/python research/optevo.py --generations 25 --pop 30
         ./venv/bin/python research/optevo.py --daemon
Отчёты:  research/results/optevo.json + panel_meta['opt'].
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
import research.optcore as op
from research.evo import LAB_IDEAS, MICRO_IDEAS

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

OPT_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'APTUSDT',
               'SEIUSDT', 'DOGEUSDT', 'BNBUSDT']
RAW_SYMBOLS = [s.replace('USDT', '/USDT:USDT') for s in OPT_SYMBOLS]
EXPIRIES = (20, 30, 60, 120, 240, 480)
PREM_KS = (0.7, 0.85, 1.0)
IDEAS = list(LAB_IDEAS) + [m.split(':', 1)[1] for m in MICRO_IDEAS]
IDEAS = [i for i in IDEAS if i != 'btc_led']

SEEDS = [
    ('APTUSDT', 15, 'mr_rsi5', 60),
    ('XRPUSDT', 60, 'mr_vwap', 120),
    ('SOLUSDT', 60, 'mr_vwap', 120),
    ('SEIUSDT', 15, 'doji_exhaust', 30),
    ('BTCUSDT', 15, 'impulse', 30),
]


def opt_genome(sym, tf, idea, expiry, prem_k=0.85, atr_min=0.03, vol_mult=1.3):
    return bt.Genome.new(sym, tf, idea, 'OPT', dict(
        expiry_min=expiry, prem_k=prem_k, atr_min_pct=atr_min,
        vol_mult=vol_mult))


class OptEvo:
    def __init__(self, pop_size=30, seed=11):
        self.rng = random.Random(seed)
        self.pop_size = pop_size
        self.hall = []
        self.gen = 0

    def score(self, views, g, lo, hi):
        st, _ = op.evaluate_option(views, g, lo, hi)
        return st

    @staticmethod
    def fit_of(st):
        return st['fitness'] if st['n'] >= 20 else -999.0

    def seed_population(self):
        pop = []
        for sym, tf, idea, E in SEEDS:
            pop.append(opt_genome(sym, tf, idea, E))
        while len(pop) < self.pop_size:
            g = opt_genome(self.rng.choice(OPT_SYMBOLS),
                           15 if self.rng.random() < 0.6 else 60,
                           self.rng.choice(IDEAS),
                           self.rng.choice(EXPIRIES),
                           self.rng.choice(PREM_KS),
                           round(self.rng.uniform(0.02, 0.08), 3),
                           round(self.rng.uniform(1.0, 2.0), 2))
            g = self.mutate(g)
            if str(g['symbol']) == 'BTCUSDT' and g['idea'] == 'btc_led':
                g['idea'] = 'impulse'
            pop.append(g)
        self.pop = pop

    def mutate(self, g):
        c = g.clone()
        r = self.rng
        p = dict(c['params'])
        if r.random() < 0.5:
            p['atr_min_pct'] = round(max(0.01, min(0.12,
                float(p['atr_min_pct']) * r.uniform(0.75, 1.3))), 3)
        if r.random() < 0.5:
            p['vol_mult'] = round(max(0.8, min(2.5,
                float(p['vol_mult']) * r.uniform(0.7, 1.4))), 2)
        if r.random() < 0.5:
            j = EXPIRIES.index(min(EXPIRIES, key=lambda e: abs(e - int(p['expiry_min']))))
            p['expiry_min'] = EXPIRIES[max(0, min(len(EXPIRIES) - 1,
                                 j + r.choice((-2, -1, 1, 2))))]
        if r.random() < 0.3:
            p['prem_k'] = r.choice(PREM_KS)
        c['params'] = p
        if r.random() < 0.25:
            c['symbol'] = r.choice(OPT_SYMBOLS)
        if r.random() < 0.25:
            c['idea'] = r.choice(IDEAS)
        if str(c['symbol']).startswith('BTC') and c['idea'] == 'btc_led':
            c['idea'] = 'impulse'
        return c

    def crossover(self, a, b):
        c = a.clone()
        c['symbol'] = self.rng.choice([a['symbol'], b['symbol']])
        c['tf'] = self.rng.choice([a['tf'], b['tf']])
        c['idea'] = self.rng.choice([a['idea'], b['idea']])
        p = dict(a['params'])
        for k in ('expiry_min', 'prem_k', 'atr_min_pct', 'vol_mult'):
            p[k] = self.rng.choice([p[k], b['params'][k]])
        c['params'] = p
        return c

    def tourn(self, scored, k=3):
        cands = self.rng.sample(scored, min(k, len(scored)))
        return max(cands, key=lambda x: x[0])[1]

    def run(self, views, lo, hi, generations, log=print):
        for gen in range(generations):
            scored = []
            for g in self.pop:
                st = self.score(views, g, lo, hi)
                scored.append((self.fit_of(st), g, st))
            scored.sort(key=lambda x: x[0], reverse=True)
            self.gen += 1
            best_fit, best_g, best_st = scored[0]
            if best_fit > -999 and (not self.hall or best_fit > self.hall[0][0]):
                self.hall.insert(0, (best_fit, best_g, dict(best_st), self.gen))
                self.hall = self.hall[:8]
                log('[opt gen %d] NEW BEST %.1fR-prem fit | %s | %s'
                    % (self.gen, best_fit, best_g.key, best_st), flush=True)
            elite = scored[:max(2, self.pop_size // 6)]
            nxt = [g.clone() for _, g, _, _ in self.hall[:3]]
            nxt += [g.clone() for _, g, _ in elite]
            while len(nxt) < self.pop_size:
                a = self.tourn(scored)
                b = self.tourn(scored)
                child = self.crossover(a, b) if self.rng.random() < 0.5 else a.clone()
                if self.rng.random() < 0.85:
                    child = self.mutate(child)
                nxt.append(child)
            self.pop = nxt


def snapshot_report(oevo, views, t_val_start, t_end):
    rep = dict(measured=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               gen=oevo.gen, pop=len(oevo.pop), hall=[])
    for fit, g, st_tr, gen_num in oevo.hall[:6]:
        stv, tr = op.evaluate_option(views, g, t_val_start, t_end)
        dig_tr = sum(1 for t in tr if t['digital_win'])
        rep['hall'].append(dict(fitness=round(fit, 1), gen=gen_num,
                                genome=g.to_json(),
                                train=dict(n=st_tr['n'], sum_bp=st_tr['sum_bp'],
                                           months_pos=st_tr.get('months_pos')),
                                val=dict(n=stv['n'], sum_bp=stv['sum_bp'],
                                         wr_vanilla=stv['wr_vanilla'],
                                         wr_digital=stv['wr_digital'],
                                         dig_wins=dig_tr),
                                prem_bp=stv.get('prem_bp')))
    rep['slippage'] = bt.slippage_report()
    rep['shadow'] = bt.shadow_report()
    return rep


def save_report(rep):
    os.makedirs(RES, exist_ok=True)
    json.dump(rep, open(os.path.join(RES, 'optevo.json'), 'w'), indent=1,
              ensure_ascii=False)
    pm = os.path.join(RES, 'panel_meta.json')
    if os.path.exists(pm) and rep['hall']:
        meta = json.load(open(pm))
        h = rep['hall'][0]
        g = h['genome']
        verdict = ('OPTION-EDGE' if h['val']['sum_bp'] > 0 and h['val']['n'] >= 10
                   else 'нет подтверждения')
        meta['opt'] = dict(
            gen=rep['gen'], pop=rep['pop'],
            best='%s %sm %s exp%dm | train %+.0fbp/n=%d | val %+.0fbp/n=%d | %s'
                 % (g['symbol'], g['tf'], g['idea'], g['params'].get('expiry_min', 60),
                    h['train']['sum_bp'], h['train']['n'],
                    h['val']['sum_bp'], h['val']['n'], verdict),
            shadow=rep['shadow'])
        json.dump(meta, open(pm, 'w'), indent=1, ensure_ascii=False)


def load_state(path):
    if os.path.exists(path):
        d = json.load(open(path))
        evo = OptEvo()
        evo.gen = d.get('gen', 0)
        evo.hall = [(h['fitness'], bt.Genome.from_json(h['genome']),
                     h['train'], h['gen']) for h in d.get('hall', [])]
        return evo
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--generations', type=int, default=25)
    ap.add_argument('--pop', type=int, default=30)
    ap.add_argument('--daemon', action='store_true')
    ap.add_argument('--cycle-sec', type=int, default=2400)
    args = ap.parse_args()

    state_path = os.path.join(RES, 'optevo_state.json')
    oevo = load_state(state_path) or OptEvo(pop_size=args.pop)
    oevo.pop_size = args.pop

    views = full = None
    while True:
        if views is None or args.daemon:
            print('[opt data] загрузка 190д × %d симв...' % len(OPT_SYMBOLS),
                  flush=True)
            t_ = time.time()
            full, raw, views = bt.load_views(days=190, symbols=RAW_SYMBOLS)
            l1 = views[15]
            t0 = int(min(x.ts[0] for x in l1.values()))
            t1 = int(max(x.ts[-1] for x in l1.values()))
            t_train_end = t1 - 14 * bt.DAY
            print('[opt data] ok %.0fs' % (time.time() - t_), flush=True)
        if not getattr(oevo, 'pop', None):
            oevo.seed_population()
        oevo.run(views, t0, t_train_end, args.generations)
        rep = snapshot_report(oevo, views, t_train_end, t1)
        save_report(rep)
        json.dump(dict(gen=oevo.gen, hall=[dict(fitness=h[0],
                                                genome=h[1].to_json(),
                                                train=h[2], gen=h[3])
                                          for h in oevo.hall]),
                  open(state_path, 'w'), indent=1, ensure_ascii=False)
        print('[optevo] gen=%d best=%s' % (oevo.gen,
              oevo.hall[0][1].key if oevo.hall else '-'), flush=True)
        if not args.daemon:
            break
        time.sleep(args.cycle_sec)


if __name__ == '__main__':
    main()
