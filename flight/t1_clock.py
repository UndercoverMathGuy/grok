"""Tier-1b: retrodict the freeze cohort (runs_clock) with the flight-time law.
Target anomalies: delay +390/+585/+1295 at fz N=50/200/500 (sublinear, big
intercept).  Decompose each arm's delay into
   d t_gen = d t_mem  +  d[ log(r_mem/r*) / lam ]
using per-run measurements; r* calibrated on the base arm only.
Committee: self-detected, largest log-gap in final-epoch energy.
"""
import numpy as np, glob
from collections import defaultdict

ROOT = '/Users/ruhaanrajadhyaksha/projects/grok'

def first_sustained(mask):
    idx = np.where(mask[:-1] & mask[1:])[0]
    return int(idx[0]) if len(idx) else -1

def committee_final(E):
    e = np.log10(np.clip(E[-1], 1e-12, None))
    order = np.argsort(e)[::-1]
    gaps = e[order[:-1]] - e[order[1:]]
    kmax = min(10, len(gaps))
    K = int(np.argmax(gaps[:kmax])) + 1
    return order[:K].tolist()

arms = defaultdict(dict)
for path in sorted(glob.glob(f'{ROOT}/runs_clock/*/*/spectra.npz')):
    arm, run = path.split('/')[-3], path.split('/')[-2]
    z = np.load(path)
    ep, E, ta, te = z['epochs'], z['energy'], z['train_acc'], z['test_acc']
    comm = committee_final(E)
    junk = [j for j in range(E.shape[1]) if j not in comm]
    r = E[:, junk].sum(1) / E.sum(1)
    im, ig = first_sustained(ta >= 0.99), first_sustained(te >= 0.9)
    if im < 0: continue
    rec = dict(ep=ep, r=r, im=im, ig=ig, K=len(comm),
               t_mem=int(ep[im]), t_gen=int(ep[ig]) if ig >= 0 else None,
               r_mem=float(r[im]))
    # lambda window starts only after BOTH memorization and unfreeze;
    # r_mem is likewise re-read at the window start (junk level when
    # cleanup actually begins), so the law clock starts at t0
    N = int(arm[2:]) if arm[:2] in ('fz', 'on') else 0
    t0 = max(rec['t_mem'], N)
    rec['t0'] = t0
    rec['r_mem'] = float(r[np.searchsorted(ep, t0)])
    iw = np.where((ep >= t0) & (ep <= t0 + 500))[0]
    rec['lam'] = None
    if len(iw) >= 5:
        sl = np.polyfit(ep[iw].astype(float), np.log(np.clip(r[iw], 1e-12, None)), 1)[0]
        if -sl > 0: rec['lam'] = -float(sl)
    arms[arm][run] = rec

# calibrate r* on base arm
base = arms['base']
r_star = float(np.median([x['r'][x['ig']] for x in base.values() if x['t_gen'] is not None]))
print(f'base arm: n={len(base)}, r* = 10^{np.log10(r_star):.2f}')

def wait(x):
    if x['lam'] is None: return None
    return max(0.0, np.log(x['r_mem'] / r_star) / x['lam'])

print(f"\n{'arm':6} {'n':>2} {'d_t_gen':>8} | {'d_t_mem':>8} {'d_wait_pred':>11} {'sum':>8} | "
      f"{'d_log10_rmem':>12} {'lam_ratio':>9} {'K_med':>5}")
b_med = {k: np.median([x[k] for x in base.values() if x['t_gen'] is not None])
         for k in ('t_mem', 't_gen', 'r_mem')}
b_lam = np.median([x['lam'] for x in base.values() if x['lam']])
b_wait = np.median([wait(x) for x in base.values() if wait(x) is not None])
for arm in ['base', 'fz50', 'fz200', 'fz500', 'on50', 'on200', 'on500']:
    xs = [x for x in arms[arm].values() if x['t_gen'] is not None]
    if not xs: print(f'{arm:6} no generalized runs'); continue
    dtg = np.median([x['t_gen'] for x in xs]) - b_med['t_gen']
    dtm = np.median([x['t_mem'] for x in xs]) - b_med['t_mem']
    ws = [wait(x) for x in xs if wait(x) is not None]
    dwait = np.median(ws) - b_wait
    dlr = np.median([np.log10(x['r_mem']) for x in xs]) - np.log10(b_med['r_mem'])
    lamr = np.median([x['lam'] for x in xs if x['lam']]) / b_lam
    K = np.median([x['K'] for x in xs])
    print(f'{arm:6} {len(xs):>2} {dtg:8.0f} | {dtm:8.0f} {dwait:11.0f} {dtm+dwait:8.0f} | '
          f'{dlr:12.2f} {lamr:9.2f} {K:5.0f}')

# per-run law accuracy pooled over all arms
pred, act = [], []
for arm, rs in arms.items():
    for x in rs.values():
        if x['t_gen'] is None or wait(x) is None: continue
        pred.append(x['t0'] + wait(x)); act.append(x['t_gen'])
pred, act = np.array(pred), np.array(act)
resid = np.log10(pred) - np.log10(act)
print(f'\npooled law on clock cohort, n={len(pred)}: Pearson(log) '
      f'{np.corrcoef(np.log10(pred),np.log10(act))[0,1]:.3f}  '
      f'R2(log) {1-np.var(resid)/np.var(np.log10(act)):.3f}  median miss x{10**np.median(np.abs(resid)):.2f}')
