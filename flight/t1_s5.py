"""Tier-1d: S5 unification + m96 transfer.
Hypothesis: the law holds everywhere; S5's wd-coupled 'selection clock' is
explained by selection COMMITTING late — during/after memorization, inside
the cleanup phase — so any dial on cleanup rate (wd) drags t_ident with it.
Statistic: t_commit = first epoch from which instantaneous top-K equals the
final committee at every later snapshot; compare t_commit/t_mem across tasks.
Also: law fit (r_mem, lambda, r_gen, prediction) per task.
"""
import numpy as np, glob
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
    return set(o[:K].tolist()), K

def t_commit(E, comm, ep):
    K = len(comm)
    ok = np.array([set(np.argsort(E[i])[::-1][:K].tolist()) == comm
                   for i in range(E.shape[0])])
    # first index from which ok holds forever after
    bad = np.where(~ok)[0]
    i0 = 0 if len(bad) == 0 else int(bad[-1]) + 1
    return int(ep[i0]) if i0 < len(ep) else None

def analyze(pattern, label, early=500):
    out = defaultdict(list)
    for path in sorted(glob.glob(pattern)):
        cell = path.split('/')[-3]
        z = np.load(path)
        ep, E, ta, te = z['epochs'], z['energy'], z['train_acc'], z['test_acc']
        comm, K = committee_final(E)
        junk = [j for j in range(E.shape[1]) if j not in comm]
        r = E[:, junk].sum(1) / E.sum(1)
        im, ig = fs(ta >= 0.99), fs(te >= 0.9)
        if im < 0: continue
        tm = int(ep[im])
        tc = t_commit(E, comm, ep)
        iw = np.where((ep >= tm) & (ep <= tm + early))[0]
        lam = None
        if len(iw) >= 5:
            sl = np.polyfit(ep[iw].astype(float), np.log(np.clip(r[iw], 1e-12, None)), 1)[0]
            if -sl > 0: lam = -float(sl)
        out[cell].append(dict(K=K, t_mem=tm, t_commit=tc,
                              t_gen=int(ep[ig]) if ig >= 0 else None,
                              r_mem=float(r[im]),
                              r_gen=float(r[ig]) if ig >= 0 else None, lam=lam))
    print(f'== {label}')
    allk = [x for xs in out.values() for x in xs if x['t_gen'] is not None]
    for cell in sorted(out):
        xs = out[cell]
        kept = [x for x in xs if x['t_gen'] is not None]
        med = lambda k, arr: (np.median([x[k] for x in arr if x[k] is not None])
                              if any(x[k] is not None for x in arr) else float('nan'))
        print(f'  {cell:18} n={len(xs)} gen={len(kept)}  K={med("K",xs):.0f}  '
              f't_mem={med("t_mem",xs):.0f}  t_commit={med("t_commit",xs):.0f}  '
              f'commit/mem={med("t_commit",xs)/max(med("t_mem",xs),1):.2f}  '
              f'log10 r_mem={np.log10(med("r_mem",xs)):.2f}  '
              f'lam/kep={1e3*med("lam",xs):.2f}  '
              f'log10 r_gen={(np.log10(med("r_gen",kept)) if kept else float("nan")):.2f}  '
              f't_gen={med("t_gen",kept) if kept else float("nan"):.0f}')
    # law fit pooled within task
    if len(allk) >= 6:
        r_star = float(np.median([x['r_gen'] for x in allk]))
        pred, act = [], []
        for x in allk:
            if not x['lam']: continue
            w = max(0.0, np.log(x['r_mem'] / r_star) / x['lam'])
            pred.append(x['t_mem'] + w); act.append(x['t_gen'])
        if len(pred) >= 6:
            lp, la = np.log10(pred), np.log10(act)
            resid = lp - la
            print(f'  law: n={len(pred)}  Pearson(log) {np.corrcoef(lp,la)[0,1]:.3f}  '
                  f'R2(log) {1-np.var(resid)/np.var(la):.3f}  '
                  f'median miss x{10**np.median(np.abs(resid)):.2f}  '
                  f'r*=10^{np.log10(r_star):.2f}')

analyze(f'{ROOT}/runs_degrok/p113_f0.3_wd*/*/spectra.npz', 'mod-add reference (degrok f0.3)')
analyze(f'{ROOT}/runs_tasks/m96_*/*/spectra.npz', 'mod-96 composite')
analyze(f'{ROOT}/runs_tasks/s5_*/*/spectra.npz', 'S5 (runs_tasks)', early=1000)
analyze(f'{ROOT}/runs_tasks_ext/s5_*/*/spectra.npz', 'S5 (ext)', early=1000)
