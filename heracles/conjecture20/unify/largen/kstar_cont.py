# Continuation in n for the slot-symmetric k* (see kstar_sym.py).  At each n, start from the previous optimum
# (same atoms; optionally insert an atom between neighbours) plus fresh structured starts.
import sys, math, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, '.')
from kstar_sym import make, run
from scipy.optimize import minimize

def polish(n, s, z, m):
    f = make(n, s)
    def fun(zz):
        v, g, _ = f(zz, m)
        if not np.isfinite(v) or g is None: return 1e6, np.zeros_like(zz)
        return -v, -g
    res = minimize(fun, z, jac=True, method='L-BFGS-B', options=dict(maxiter=6000, gtol=1e-12))
    v, _, info = f(res.x, m, False)
    return (v if info is not None and info['num'] > 0 else -np.inf), res.x, info

def insert_atom(z, m):
    rho, al, ph = z[:m], z[m:2*m], z[2*m:]
    o = np.argsort(ph); rho, al, ph = rho[o], al[o], ph[o]
    outs = []
    # add at either end (extrapolate) or in the middle
    for pos in ['lo', 'hi']:
        if pos == 'lo':
            ph2 = np.concatenate([[2*ph[0]-ph[1]], ph]); al2 = np.concatenate([[2*al[0]-al[1]], al]); rho2 = np.concatenate([[rho[0]-1.0], rho])
        else:
            ph2 = np.concatenate([ph, [2*ph[-1]-ph[-2]]]); al2 = np.concatenate([al, [2*al[-1]-al[-2]]]); rho2 = np.concatenate([rho, [rho[-1]-1.0]])
        outs.append(np.concatenate([rho2, al2, ph2]))
    return outs

if __name__ == "__main__":
    n0, n1 = [int(v) for v in sys.argv[1].split(':')]
    bval = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    starts = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    cands = {}   # m -> z
    for n in range(n0, n1 + 1):
        s = (n + 1) / bval
        newc = {}
        # continue every stored candidate, and its one-atom extensions
        for m, z in list(cands.items()):
            v, zz, info = polish(n, s, z, m)
            if v > newc.get(m, (-np.inf,))[0]: newc[m] = (v, zz)
            for z2 in insert_atom(z, m):
                v, zz, info = polish(n, s, z2, m + 1)
                if v > newc.get(m + 1, (-np.inf,))[0]: newc[m + 1] = (v, zz)
        # fresh starts around m ~ 0.75 n
        for m in sorted(set([max(2, int(0.65 * n)), max(2, int(0.75 * n)), max(2, int(0.85 * n))])):
            v, zz, info = run(n, m, s, starts, seed=31 * n + m)
            if v > newc.get(m, (-np.inf,))[0]: newc[m] = (v, zz)
        # keep the best 4 widths
        keep = sorted(newc.items(), key=lambda kv: -kv[1][0])[:4]
        cands = {m: vz[1] for m, vz in keep}
        mb, (vb, zb) = keep[0]
        f = make(n, s); _, _, info = f(zb, mb, False)
        L = np.exp(zb[:mb]).sum()
        print(f"n={n} s={s:.2f} k*_sym>={vb:.4f} m={mb} (others: " + ", ".join(f"{m}:{vz[0]:.4f}" for m, vz in keep[1:]) +
              f") |A|/L={abs(info['A'])/L:.3f} IF/L={info['IF']/L:.4f} E/|A|^2={info['E']/abs(info['A'])**2:.3f}", flush=True)
        np.save(f"kstar_opt_n{n}_b{bval}.npy", np.concatenate([[mb], zb]))
