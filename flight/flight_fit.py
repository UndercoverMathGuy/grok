"""Flight-time law fit on degrok 80 runs.

Model: after memorization, non-committee energy share r(t)=1-share decays
exponentially at rate lam; generalization fires when r crosses a universal
threshold r*.  Predicted wait = log(r_mem/r*) / lam.

Non-circular test: lam fitted ONLY on [t_mem, t_mem+EARLY] window; r* is a
single global constant (median r at t_gen over OTHER cells' runs would be
stricter; here global median, reported both ways).
"""
import numpy as np, json, glob, re
from collections import defaultdict

ROOT = '/Users/ruhaanrajadhyaksha/projects/grok'
EARLY = 500  # epochs after t_mem used to fit the rate

d = json.load(open(f'{ROOT}/notes/degrok_summary.json'))
meta = {}
for r in d['runs']:
    key = (str(r['p']), str(float(r['frac'])), str(float(r['wd'])), r['run'])
    comm = json.loads(r['committee']) if isinstance(r['committee'], str) else r['committee']
    meta[key] = comm

rows = []
for path in sorted(glob.glob(f'{ROOT}/runs_degrok/*/*/spectra.npz')):
    rd = path.rsplit('/', 1)[0]
    cell, run = rd.split('/')[-2], rd.split('/')[-1]
    m = re.match(r'p(\d+)_f([\d.]+)_wd([\d.]+)', cell)
    key = (m.group(1), str(float(m.group(2))), str(float(m.group(3))), run)
    if key not in meta:
        continue
    comm = meta[key]
    z = np.load(path)
    ep, E, ta, te = z['epochs'], z['energy'], z['train_acc'], z['test_acc']
    Ec = E[:, [k - 1 for k in comm]].sum(1)
    r = 1.0 - Ec / E.sum(1)
    # memorization and generalization times (sustained x2 snapshots)
    mem = (ta >= 0.99)
    gen = (te >= 0.9)
    def first_sustained(mask):
        idx = np.where(mask[:-1] & mask[1:])[0]
        return int(idx[0]) if len(idx) else -1
    im, ig = first_sustained(mem), first_sustained(gen)
    if im < 0:
        continue
    t_mem = int(ep[im])
    t_gen = int(ep[ig]) if ig >= 0 else None       # None = censored
    r_mem = float(r[im])
    r_gen = float(r[ig]) if ig >= 0 else None
    # early-window rate fit on log r
    iw = np.where((ep >= t_mem) & (ep <= t_mem + EARLY))[0]
    lam = None
    if len(iw) >= 5:
        x, y = ep[iw].astype(float), np.log(np.clip(r[iw], 1e-12, None))
        b = np.polyfit(x, y, 1)[0]
        lam = -float(b)  # decay rate per epoch
    rows.append(dict(cell=cell, run=run, p=int(m.group(1)), frac=float(m.group(2)),
                     wd=float(m.group(3)), t_mem=t_mem, t_gen=t_gen,
                     r_mem=r_mem, r_gen=r_gen, lam=lam))

kept = [x for x in rows if x['t_gen'] is not None]
cens = [x for x in rows if x['t_gen'] is None]
print(f'{len(rows)} runs loaded, {len(kept)} generalized, {len(cens)} censored')

# --- 1. threshold universality ---
rg = np.array([x['r_gen'] for x in kept])
rm = np.array([x['r_mem'] for x in kept])
print('\n[1] threshold universality: log10 r_gen  median %.2f  IQR [%.2f, %.2f]  full [%.2f, %.2f]'
      % (np.median(np.log10(rg)), *np.percentile(np.log10(rg), [25, 75]),
         np.log10(rg.min()), np.log10(rg.max())))
print('    starting junk:            log10 r_mem  median %.2f  full [%.2f, %.2f]'
      % (np.median(np.log10(rm)), np.log10(rm.min()), np.log10(rm.max())))

# --- 2. r_mem vs frac ; lam vs wd ---
by_frac = defaultdict(list); by_cell_lam = defaultdict(list)
for x in kept:
    by_frac[x['frac']].append(x['r_mem'])
    if x['lam'] is not None:
        by_cell_lam[(x['p'], x['frac'], x['wd'])].append(x['lam'])
print('\n[2] median r_mem by frac:', {f: round(float(np.median(v)), 4) for f, v in sorted(by_frac.items())})
print('    median lam by cell (per kilo-epoch):')
for k in sorted(by_cell_lam):
    print('     p%d f%.1f wd%.0f: %.3f' % (*k, 1e3 * np.median(by_cell_lam[k])))

# --- 3. the law: predict wait from (r_mem, lam_early, global r*) ---
r_star = float(np.median(rg))
pred, act = [], []
for x in kept:
    if x['lam'] is None or x['lam'] <= 0:
        continue
    w = max(0.0, np.log(x['r_mem'] / r_star) / x['lam'])
    pred.append(x['t_mem'] + w)
    act.append(x['t_gen'])
pred, act = np.array(pred), np.array(act)
lp, la = np.log10(pred), np.log10(act)
rho = np.corrcoef(lp, la)[0, 1]
def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])
sp_c = spearman(pred, act)
resid = lp - la
print(f'\n[3] flight-time law, n={len(pred)}: Pearson(log t_pred, log t_gen) = {rho:.3f}'
      f'  Spearman = {sp_c:.3f}')
print(f'    median |log10 ratio| = {np.median(np.abs(resid)):.3f} '
      f'(x{10**np.median(np.abs(resid)):.2f}), R2(log) = {1 - np.var(resid)/np.var(la):.3f}')
print(f'    actual t_gen spread: x{act.max()/act.min():.0f}')
# within-cell spearman
cells = defaultdict(list)
for x, p_, a_ in zip([x for x in kept if x['lam'] and x['lam'] > 0], pred, act):
    cells[x['cell']].append((p_, a_))
wc = [spearman(np.array([q[0] for q in v]), np.array([q[1] for q in v]))
      for v in cells.values() if len(v) >= 4]
print(f'    within-cell Spearman: median {np.median(wc):.2f} across {len(wc)} cells')

# --- 4. censored runs: does the law predict censoring? ---
for x in cens:
    if x['lam'] and x['lam'] > 0:
        w = x['t_mem'] + np.log(x['r_mem'] / r_star) / x['lam']
        print(f'[4] censored {x["cell"]}/{x["run"]}: predicted t_gen {w:.0f} (budget 10000)')
