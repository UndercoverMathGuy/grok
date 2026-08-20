"""Every-run evidence sweep for the lambda decomposition.

Claims under test (lambda = kappa * 2*lr*wd * Deff + lambda_0):
  A. Instrument: lambda_share ~= g_c - g_j  (differential growth identity;
     exact form is lambda = g_tot - g_j, approaches g_c - g_j as share -> 1)
  B. No direct deletion: junk energy does NOT decay at the naive wd rate;
     mostly it grows (g_j > 0) while its share falls.
  C. Treadmill: lambda scales with wd above a floor (paired cells).
  D. Floor: at wd <= 0.3 lambda is wd-independent (lambda_0).
  E. Exchange rate: kappa*Deff = lambda / (2*lr*wd) is task-dependent,
     ordered add >> s5.
Natural cohorts only: degrok, degrok2, dial, race, clock/base, tasks(+ext).
Committee: largest log-gap in final-epoch energy (uniform instrument).
"""
import numpy as np, json, glob, os
from collections import defaultdict

ROOT = '/Users/ruhaanrajadhyaksha/projects/grok'

def fs(m):
    i = np.where(m[:-1] & m[1:])[0]
    return int(i[0]) if len(i) else -1

def committee_final(E):
    e = np.log10(np.clip(E[-1], 1e-12, None))
    o = np.argsort(e)[::-1]
    g = e[o[:-1]] - e[o[1:]]
    K = int(np.argmax(g[:min(10, len(g))])) + 1
    return o[:K].tolist()

PATTERNS = [
    ('degrok',  f'{ROOT}/runs_degrok/*/*/spectra.npz'),
    ('degrok2', f'{ROOT}/runs_degrok2/*/*/spectra.npz'),
    ('dial',    f'{ROOT}/runs_dial/*/*/spectra.npz'),
    ('race',    f'{ROOT}/runs_race/*/*/spectra.npz'),
    ('clock',   f'{ROOT}/runs_clock/base/*/spectra.npz'),
    ('tasks',   f'{ROOT}/runs_tasks/*/*/spectra.npz'),
    ('tasksx',  f'{ROOT}/runs_tasks_ext/*/*/spectra.npz'),
]

rows = []
for cohort, pat in PATTERNS:
    for path in sorted(glob.glob(pat)):
        rd = os.path.dirname(path)
        try:
            cfg = json.load(open(rd + '/config.json'))
        except FileNotFoundError:
            cfg = {}
        cell = rd.split('/')[-2]
        wd = cfg.get('weight_decay')
        if wd is None and 'wd' in cell:
            wd = float(cell.split('wd')[-1])
        lr = cfg.get('lr', 1e-3)
        fn = cfg.get('fn_name', 's5' if cell.startswith('s5') else 'add')
        p = cfg.get('p', 0)
        task = f'{fn}{p}' if fn != 's5' else 's5'
        z = np.load(path)
        ep, E, ta, te = z['epochs'], z['energy'], z['train_acc'], z['test_acc']
        comm = committee_final(E)
        junk = [j for j in range(E.shape[1]) if j not in comm]
        Ec, Ej, Et = E[:, comm].sum(1), E[:, junk].sum(1), E.sum(1)
        im, ig = fs(ta >= 0.99), fs(te >= 0.9)
        if im < 0 or ig < 0 or ig - im < 5:
            continue
        x = ep[im:ig + 1].astype(float)
        lg = lambda v: np.log(np.clip(v[im:ig + 1], 1e-12, None))
        lam = -np.polyfit(x, lg(Ej / Et), 1)[0] * 1e3     # share rate, /kep
        g_c = np.polyfit(x, lg(Ec), 1)[0] * 1e3
        g_j = np.polyfit(x, lg(Ej), 1)[0] * 1e3
        rows.append(dict(cohort=cohort, cell=cell, task=task, wd=float(wd),
                         lr=lr, width=cfg.get('d_mlp', 0), K=len(comm),
                         frac=cfg.get('frac_train', 0), t_mem=int(ep[im]),
                         t_gen=int(ep[ig]), wait=int(ep[ig] - ep[im]),
                         lam=lam, g_c=g_c, g_j=g_j,
                         naive=2 * lr * float(wd) * 1e3,
                         r_gen=float((Ej / Et)[ig])))
print(f'{len(rows)} generalized runs with measurable waits')

R = rows
lam = np.array([r['lam'] for r in R]); gc = np.array([r['g_c'] for r in R])
gj = np.array([r['g_j'] for r in R]); naive = np.array([r['naive'] for r in R])

# A. identity
diff = gc - gj
print(f'\nA. corr(lambda, g_c - g_j) = {np.corrcoef(lam, diff)[0,1]:.3f}; '
      f'median |lambda-(g_c-g_j)| = {np.median(np.abs(lam-diff)):.2f} /kep '
      f'(median lambda {np.median(lam):.2f})')

# B. junk deletion
grow = (gj > 0).mean()
del_naive = (gj < -naive).mean()
print(f'B. junk GROWS during wait in {100*grow:.0f}% of runs; '
      f'junk decays at >= naive 2*lr*wd rate in only {100*del_naive:.0f}%')
for wd in sorted({r['wd'] for r in R}):
    sub = [r for r in R if r['wd'] == wd]
    print(f'   wd={wd}: n={len(sub)}  g_j>0 in {100*np.mean([r["g_j"]>0 for r in sub]):.0f}%  '
          f'median g_j {np.median([r["g_j"] for r in sub]):+.2f} vs naive deletion -{2*1e3*1e-3*wd:.1f}')

# C. paired wd scaling within (cohort, task, width, frac)
print('\nC. paired lambda ratio across wd (same task/width/frac):')
groups = defaultdict(dict)
for r in R:
    groups[(r['cohort'], r['task'], r['width'], r['frac'])].setdefault(r['wd'], []).append(r['lam'])
ratios = []
for k, d in sorted(groups.items()):
    wds = sorted(d)
    if len(wds) >= 2:
        lo, hi = wds[0], wds[-1]
        ratio = np.median(d[hi]) / max(np.median(d[lo]), 1e-9)
        ratios.append((k, lo, hi, ratio, hi / lo))
        print(f'   {k[0]:8}{k[1]:7} w{k[2]:<5} f{k[3]:<5} wd {lo}->{hi}: '
              f'lambda x{ratio:5.2f} (wd x{hi/lo:.0f})')
med_r = np.median([x[3] for x in ratios if x[4] == 3.0])
print(f'   median lambda-ratio where wd x3: x{med_r:.2f}')

# D. floor at low wd (dial + race)
print('\nD. low-wd floor, lambda medians (/kep):')
for cohort in ('dial', 'race'):
    for wd in (0.1, 0.3, 1.0, 3.0):
        v = [r['lam'] for r in R if r['cohort'] == cohort and r['wd'] == wd]
        if v: print(f'   {cohort} wd={wd}: {np.median(v):.3f} (n={len(v)})  naive={2e0*wd:.1f}')

# E. exchange rate by task (wd >= 1 only, driven regime)
print('\nE. kappa*Deff = lambda/(2*lr*wd) by task (wd>=1):')
for task in sorted({r['task'] for r in R}):
    v = [r['lam'] / r['naive'] for r in R if r['task'] == task and r['wd'] >= 1]
    if v: print(f'   {task:8}: median {np.median(v):.3f}  IQR [{np.percentile(v,25):.3f},{np.percentile(v,75):.3f}]  n={len(v)}')

# bonus: r* by task
print('\nr* (log10 r_gen) by task:')
for task in sorted({r['task'] for r in R}):
    v = np.log10([r['r_gen'] for r in R if r['task'] == task])
    print(f'   {task:8}: median {np.median(v):.2f}  n={len(v)}')

json.dump(rows, open(f'{ROOT}/notes/lambda_evidence.json', 'w'), indent=1)
print(f'\nwrote notes/lambda_evidence.json ({len(rows)} rows)')
