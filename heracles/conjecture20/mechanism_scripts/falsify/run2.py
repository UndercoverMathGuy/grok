# c* search with block seeds: IF part (programme support y*, v=0) + cheap block P (v=cap*a) with flat-ish phases,
# plus the run.py seeds.  Each seed: gradient (crit) -> LP bisection polish (fixed phases) -> gradient -> polish.
import numpy as np, sys, time, math, torch, os
from sympy import legendre_symbol
from gget import gamma_table
from relmodel import Model, optimise, lp_polish, crit_polish
from seeds import seeds, fold, mk
torch.set_num_threads(1)
s = 1.5; CS = 1 - (2 / 3) ** 0.5; CA = 1 / (2 * 3 ** 0.5); PAIR = 1.5 ** 0.5
p = int(sys.argv[1]); nrand = int(sys.argv[2]) if len(sys.argv) > 2 else 1
T = gamma_table()[p]; G = T['Gamma']; M = Model(p, s, CS, G); F = M.F; L = math.log(p)
rng = np.random.default_rng(10 * p + 7)
Sy = np.array(T['S']) - 1; y = np.array(T['y'])
chi = np.array([int(legendre_symbol(l, p)) for l in range(1, F + 1)])
thleg = np.where(chi > 0, 0.0, np.pi) if p % 4 == 1 else -chi * np.pi / 2
SD = []   # (name, S, a0, v0, th0)
for name, S, A, TH in seeds(p, rng, (T['S'], y), nrand=nrand):
    SD.append((name, np.array(S), A, M.cap * A * 0.9, TH))
rest = np.setdiff1d(np.arange(F), Sy)
for Mm in sorted(set([int(round(x * L)) for x in [2, 4, 8]] + [len(rest)])):
    Mm = min(Mm, len(rest))
    for u in [0.5, 2.0]:
        for kind in ['rand', 'AP', 'leg']:
            if kind == 'AP':
                d = int(rng.integers(1, p)); l0 = int(rng.integers(1, p))
                im = fold([(l0 + k * d) % p for k in range(3 * Mm)], p)
                P = np.array([i for i in im if i not in set(Sy)])[:Mm]
                ph = {i: im[i] * math.pi * k * k / Mm for k, i in enumerate(P)}
                th = rng.uniform(0, 2 * np.pi, F)
                for i, v in ph.items(): th[i] = v
            else:
                P = np.sort(rng.choice(rest, Mm, replace=False))
                th = thleg.copy() if kind == 'leg' else rng.uniform(0, 2 * np.pi, F)
            A = np.zeros(F); A[Sy] = y
            b = (u * np.sum(y ** M.r) / len(P)) ** s; A[P] = b
            V = np.zeros(F); V[P] = M.cap * b * 0.95; V[Sy] = 1e-4 * y
            SD.append((f'IF+{kind}{len(P)}u{u}', np.sort(np.concatenate([Sy, P])), A, V, th))
t0 = time.time(); best = (9, None, None)
for name, S, A, V, TH in SD:
    a, v, th = optimise(M, S, A + 1e-12, np.maximum(V, 1e-12), TH, mode='crit')
    c1, _ = M.cstar(a, v, th)
    try:
        c2, a, v = crit_polish(M, a, v, th)
        S2 = np.nonzero(a > 1e-9 * a.max())[0]
        a, v, th = optimise(M, S2, a, np.maximum(v, 1e-12 * a.max()), th, mode='crit', stages=(0.003, 0.001, 0.0003))
        c3, a, v = crit_polish(M, a, v, th)
    except Exception as e:
        c3 = M.cstar(a, v, th)[0]
    inf = M.info(a, v, th)
    msg = f'{name:22s} c*: grad {c1:.4f} -> polished {c3:.4f}  supp={inf["supp"]} cheap={inf["cheap"]}  ratio@sharp={M.ratio(a, v, th, CS):.4f} @A={M.ratio(a, v, th, CA):.4f}'
    print(msg + f'  [{time.time() - t0:.0f}s]', flush=True)
    if c3 < best[0]: best = (c3, name, (a, v, th))
c, name, (a, v, th) = best
cov, B, osc = M.parts(a, v, th); nrm = M.norm(a)
cd = (nrm - cov + osc / 2) / B; dd = np.argmax(cd)
ch = v > 1e-6 * a.max()
lam = (osc[dd] / 2) / np.sqrt(np.sum(v ** 2 * M.Knp[dd]))
print(f'== p={p} Gamma={G:.6f} BEST c*={c:.4f} from {name}; supp={np.sum(a > 1e-6 * a.max())} cheap={ch.sum()} '
      f'IF-cov/norm at worst delta={cov[dd] / nrm:.3f} covmin/norm={cov.min() / nrm:.3f} lam={lam:.3f}', flush=True)
os.makedirs('res', exist_ok=True)
np.save(f'res/run2_p{p}.npy', dict(c=c, name=name, a=a, v=v, th=th), allow_pickle=True)
