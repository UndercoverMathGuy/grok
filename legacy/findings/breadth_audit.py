"""Breadth audit: committee/floor/menu/additive stats on EVERY grokked
spectra-logged run in runs/, grouped by family. Fast: relM_equal only (no
LP), cached nulls per (p, K)."""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path('/Users/ruhaanrajadhyaksha/projects/grok')
sys.path.insert(0, str(ROOT / 'findings'))
sys.path.insert(0, str(ROOT / 'scripts'))
from mask_lottery import committee_from_coeffs
from margin_analysis import relM_equal

rng = np.random.default_rng(0)
NULLS = {}
def pctile(comm, p, n=3000):
    K = len(comm)
    if (p, K) not in NULLS:
        nf = p // 2
        draws = [relM_equal(sorted(rng.choice(np.arange(1, nf + 1), K,
                 replace=False).tolist()), p) for _ in range(n)]
        NULLS[(p, K)] = np.sort(draws)
    null = NULLS[(p, K)]
    return 100.0 * np.searchsorted(null, relM_equal(sorted(comm), p)) / len(null)

def fold(x, p):
    x %= p
    return min(x, p - x)

def nviol(S, p):
    S = sorted(S); ss = set(S); c = 0
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            if fold(S[i] + S[j], p) in ss or fold(S[i] - S[j], p) in ss:
                c += 1
    return c

# expected violations under uniform null, per (p, K), from the same draws
EV = {}
def exp_viol(p, K, n=1500):
    if (p, K) not in EV:
        nf = p // 2
        EV[(p, K)] = np.mean([nviol(rng.choice(np.arange(1, nf + 1), K,
                              replace=False).tolist(), p) for _ in range(n)])
    return EV[(p, K)]

NATURAL = {'og_seed0', 'seed0', 'seed1', 'seed2', 'p-113', 'p-127', 'p-157'}
SELECTION = NATURAL | {'epsfloor', 'orthWE', 'phase2-noise', 'phase2-noise2',
                       'phase2-tilt', 'phase2-probes', 'eff-A', 'eff-B',
                       'eff-C', 'eff-D', 'eff-E', 'eff-G',
                       'surgery', 'surgery2', 'transplant'}

rows = []
for cj in sorted((ROOT / 'runs').rglob('config.json')):
    d = cj.parent
    fam = str(d.relative_to(ROOT / 'runs')).split('/')[0]
    if fam not in SELECTION or not (d / 'spectra.npz').exists():
        continue
    c = json.loads(cj.read_text())
    z = np.load(d / 'spectra.npz')
    acc = float(z['test_acc'][-1])
    if acc < 0.99:
        rows.append(dict(dir=str(d.relative_to(ROOT/'runs')), fam=fam, grok=False))
        continue
    p = c['p']
    comm = committee_from_coeffs(z['coeffs'][-1])
    i3k = int(np.argmin(np.abs(z['epochs'] - 3000)))
    menu8 = set((np.argsort(np.abs(z['coeffs'][i3k]))[::-1][:8] + 1).tolist())
    rows.append(dict(
        dir=str(d.relative_to(ROOT / 'runs')), fam=fam, grok=True, p=p,
        ds=c.get('data_seed'), iseed=c.get('init_seed'), K=len(comm),
        pct=pctile(comm, p), nv=nviol(comm, p), ev=exp_viol(p, len(comm)),
        closed=set(comm) <= menu8, natural=fam in NATURAL))

ok = [r for r in rows if r['grok']]
print(f"spectra-logged selection runs: {len(rows)}  grokked: {len(ok)}  "
      f"(non-grokked: {[r['dir'] for r in rows if not r['grok']]})")
print(f"masks (p, ds): {len({(r['p'], r['ds']) for r in ok})}  "
      f"init seeds: {len({r['iseed'] for r in ok})}  "
      f"primes: {sorted({r['p'] for r in ok})}")

def agg(sub, label):
    if not sub:
        return
    pcts = np.array([r['pct'] for r in sub])
    below = int((pcts < 25).sum())
    obs_v = sum(r['nv'] for r in sub)
    exp_v = sum(r['ev'] for r in sub)
    closed = sum(r['closed'] for r in sub)
    print(f"{label:<28} n={len(sub):>3}  floor: {below} below 25th pct "
          f"(min {pcts.min():5.1f}, mean {pcts.mean():5.1f})  "
          f"additive: {obs_v} obs vs {exp_v:.1f} exp  "
          f"menu-closed(top8@3k): {closed}/{len(sub)}")

agg(ok, 'ALL grokked')
agg([r for r in ok if r['natural']], 'natural dynamics')
agg([r for r in ok if not r['natural']], 'intervention families')
for fam in sorted({r['fam'] for r in ok}):
    agg([r for r in ok if r['fam'] == fam], f'  {fam}')

pcts = np.array([r['pct'] for r in ok])
n = len(ok)
print(f"\nfloor binomial: P(0 of {n} below 25th | margin irrelevant) = 0.75^{n} "
      f"= {0.75**n:.2e}" if (pcts < 25).sum() == 0 else
      f"\nfloor: {(pcts<25).sum()}/{n} below 25th pctile")
viol_runs = [(r['dir'], r['nv'], r['pct']) for r in ok if r['nv'] > 0]
print(f"final committees with additive violations: {len(viol_runs)}: {viol_runs}")
low = sorted(ok, key=lambda r: r['pct'])[:5]
print('lowest percentiles:', [(r['dir'], round(r['pct'], 1)) for r in low])
