"""Tier-1a: harden the flight-time law on degrok.
(i) leave-one-cell-out r*   (ii) band-fit lambda (clean physics numbers)
(iii) alternative order parameter (max-junk interference ratio) — does the
threshold spread shrink?
"""
import numpy as np, json, glob, re
from collections import defaultdict

ROOT = '/Users/ruhaanrajadhyaksha/projects/grok'
d = json.load(open(f'{ROOT}/notes/degrok_summary.json'))
meta = {}
for r in d['runs']:
    key = (str(r['p']), str(float(r['frac'])), str(float(r['wd'])), r['run'])
    meta[key] = json.loads(r['committee']) if isinstance(r['committee'], str) else r['committee']

def first_sustained(mask):
    idx = np.where(mask[:-1] & mask[1:])[0]
    return int(idx[0]) if len(idx) else -1

runs = []
for path in sorted(glob.glob(f'{ROOT}/runs_degrok/*/*/spectra.npz')):
    rd = path.rsplit('/', 1)[0]
    cell, run = rd.split('/')[-2], rd.split('/')[-1]
    m = re.match(r'p(\d+)_f([\d.]+)_wd([\d.]+)', cell)
    key = (m.group(1), str(float(m.group(2))), str(float(m.group(3))), run)
    if key not in meta: continue
    comm = [k - 1 for k in meta[key]]
    z = np.load(path)
    ep, E, ta, te = z['epochs'], z['energy'], z['train_acc'], z['test_acc']
    junk = [j for j in range(E.shape[1]) if j not in comm]
    Ec, Ej = E[:, comm], E[:, junk]
    r = Ej.sum(1) / E.sum(1)
    # alt order parameter: strongest junk mode vs weakest committee mode
    rho = Ej.max(1) / Ec.min(1)
    im, ig = first_sustained(ta >= 0.99), first_sustained(te >= 0.9)
    if im < 0: continue
    runs.append(dict(cell=cell, frac=float(m.group(2)), wd=float(m.group(3)),
                     ep=ep, r=r, rho=rho, im=im, ig=ig,
                     t_mem=int(ep[im]), t_gen=int(ep[ig]) if ig >= 0 else None))

kept = [x for x in runs if x['t_gen'] is not None]
print(f'{len(runs)} runs, {len(kept)} generalized')

# ---------- (iii) threshold spread: r vs rho ----------
for name in ('r', 'rho'):
    v = np.log10([x[name][x['ig']] for x in kept])
    print(f'[iii] log10 {name}_gen: median {np.median(v):.2f}  IQR {np.percentile(v,75)-np.percentile(v,25):.2f} dex  full {v.max()-v.min():.2f} dex')

# ---------- (ii) band-fit lambda: from t_mem to first crossing of 3*r_star ----------
R_STAR = 10 ** -2.39
lam_band = {}
for x in runs:
    r, ep, im = x['r'], x['ep'], x['im']
    below = np.where(r[im:] < 3 * R_STAR)[0]
    iend = im + (int(below[0]) if len(below) else len(r) - 1 - im)
    if iend - im < 4:
        x['lam_b'] = None; continue
    sl = np.polyfit(ep[im:iend + 1].astype(float), np.log(np.clip(r[im:iend + 1], 1e-12, None)), 1)[0]
    x['lam_b'] = -float(sl)
    lam_band[(x['cell'])] = lam_band.get(x['cell'], []) + [x['lam_b']]
print('\n[ii] band-fit lambda (per kilo-epoch), cell medians:')
for c in sorted(lam_band):
    print(f'   {c}: {1e3*np.median(lam_band[c]):.3f}   (naive 2*lr*wd = {2*0.001*float(c.split("wd")[1])*1e3:.1f})')

# ---------- (i) LOO r* prediction ----------
EARLY = 500
r_gen_by_cell = defaultdict(list)
for x in kept:
    r_gen_by_cell[x['cell']].append(x['r'][x['ig']])
pred, act = [], []
for x in kept:
    ep, r, im = x['ep'], x['r'], x['im']
    iw = np.where((ep >= x['t_mem']) & (ep <= x['t_mem'] + EARLY))[0]
    if len(iw) < 5: continue
    lam = -np.polyfit(ep[iw].astype(float), np.log(np.clip(r[iw], 1e-12, None)), 1)[0]
    if lam <= 0: continue
    others = [v for c, vs in r_gen_by_cell.items() if c != x['cell'] for v in vs]
    r_loo = float(np.median(others))
    w = max(0.0, np.log(r[im] / r_loo) / lam)
    pred.append(x['t_mem'] + w); act.append(x['t_gen'])
pred, act = np.array(pred), np.array(act)
lp, la = np.log10(pred), np.log10(act)
resid = lp - la
def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])
print(f'\n[i] LOO-r* law, n={len(pred)}: Pearson(log) {np.corrcoef(lp,la)[0,1]:.3f}  '
      f'Spearman {spearman(pred,act):.3f}  R2(log) {1-np.var(resid)/np.var(la):.3f}  '
      f'median miss x{10**np.median(np.abs(resid)):.2f}')
