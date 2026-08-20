"""Tier-1c: flight-time law across runs_dial (6 widths x wd {0.1,0.3,1,3}).
Questions: (1) does one r* hold across 32x width; (2) lambda vs wd and width;
(3) law R2 on generalized runs; (4) censoring prediction (t_pred > budget?).
Committees from notes/race_summary.json.  Grid = 100 ep, budget 50k.
"""
import numpy as np, json, glob
from collections import defaultdict

ROOT = '/Users/ruhaanrajadhyaksha/projects/grok'
race = json.load(open(f'{ROOT}/notes/race_summary.json'))
comm_by_run = {r['run']: r['committee'] for r in race['runs']}

def first_sustained(mask):
    idx = np.where(mask[:-1] & mask[1:])[0]
    return int(idx[0]) if len(idx) else -1

EARLY = 1000
rows = []
for path in sorted(glob.glob(f'{ROOT}/runs_dial/*/*/spectra.npz')):
    rd = path.rsplit('/', 1)[0]
    key = '/'.join(rd.split('/')[-2:])
    if key not in comm_by_run: continue
    comm = [k - 1 for k in comm_by_run[key]]
    cell = rd.split('/')[-2]
    width = int(cell.split('_')[0][1:]); wd = float(cell.split('wd')[1])
    z = np.load(path)
    ep, E, ta, te = z['epochs'], z['energy'], z['train_acc'], z['test_acc']
    junk = [j for j in range(E.shape[1]) if j not in comm]
    r = E[:, junk].sum(1) / E.sum(1)
    im, ig = first_sustained(ta >= 0.99), first_sustained(te >= 0.9)
    if im < 0: continue
    t_mem = int(ep[im])
    iw = np.where((ep >= t_mem) & (ep <= t_mem + EARLY))[0]
    lam = None
    if len(iw) >= 5:
        sl = np.polyfit(ep[iw].astype(float), np.log(np.clip(r[iw], 1e-12, None)), 1)[0]
        if -sl > 0: lam = -float(sl)
    rows.append(dict(cell=cell, width=width, wd=wd, t_mem=t_mem,
                     t_gen=int(ep[ig]) if ig >= 0 else None,
                     r_mem=float(r[im]),
                     r_gen=float(r[ig]) if ig >= 0 else None, lam=lam))

kept = [x for x in rows if x['t_gen'] is not None]
cens = [x for x in rows if x['t_gen'] is None]
print(f'{len(rows)} runs matched, {len(kept)} generalized, {len(cens)} censored')

# (1) r* by width
print('\n[1] log10 r_gen by width (median [IQR]):')
by_w = defaultdict(list)
for x in kept: by_w[x['width']].append(np.log10(x['r_gen']))
for w in sorted(by_w):
    v = np.array(by_w[w])
    print(f'   w{w:5}: {np.median(v):6.2f}  [{np.percentile(v,25):.2f},{np.percentile(v,75):.2f}]  n={len(v)}')

# (2) lambda by wd (pooled over widths), and by width at wd=1
print('\n[2] median lambda (/kep) by wd:',
      {wd: round(1e3 * float(np.median([x['lam'] for x in kept if x['wd'] == wd and x['lam']])), 2)
       for wd in sorted({x['wd'] for x in kept})})
print('    median lambda (/kep) by width at wd>=1:',
      {w: round(1e3 * float(np.median([x['lam'] for x in kept if x['width'] == w and x['wd'] >= 1 and x['lam']])), 2)
       for w in sorted(by_w)})

# (3) law on generalized runs — r* global median
r_star = float(np.median([x['r_gen'] for x in kept]))
pred, act = [], []
for x in kept:
    if not x['lam']: continue
    w = max(0.0, np.log(x['r_mem'] / r_star) / x['lam'])
    pred.append(x['t_mem'] + w); act.append(x['t_gen'])
pred, act = np.array(pred), np.array(act)
lp, la = np.log10(pred), np.log10(act)
resid = lp - la
def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])
print(f'\n[3] law, n={len(pred)}: Pearson(log) {np.corrcoef(lp,la)[0,1]:.3f}  '
      f'Spearman {spearman(pred,act):.3f}  R2(log) {1-np.var(resid)/np.var(la):.3f}  '
      f'median miss x{10**np.median(np.abs(resid)):.2f}  (t_gen spread x{act.max()/act.min():.0f})')

# (4) censoring: predict t_gen for ALL runs, classify vs 50k budget
BUDGET = 50000
tp = fp = tn = fn = 0
miss_lam = 0
for x in rows:
    if not x['lam']: miss_lam += 1; continue
    w = max(0.0, np.log(x['r_mem'] / r_star) / x['lam'])
    p_cens = (x['t_mem'] + w) > BUDGET
    a_cens = x['t_gen'] is None
    tp += p_cens and a_cens; fn += (not p_cens) and a_cens
    fp += p_cens and (not a_cens); tn += (not p_cens) and (not a_cens)
print(f'\n[4] censoring prediction (budget 50k): {tp}/{tp+fn} censored called, '
      f'{fp} false alarms out of {fp+tn} generalized  (no-lam skipped: {miss_lam})')
