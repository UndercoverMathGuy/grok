"""analyse_junk.py -- anatomy of the ascent optima that beat the schoolbook symmetric ceiling (junk_ascent.py)."""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import gt_core as gc
from junk_ascent import ascent, ceilings
from verify_fourier import irreps_of, fhat

def anatomy(gname, m, seed, steps=6000):
    G = gc.get(gname); N = G.N; ce = ceilings(G, 3, 2.0); gs = ce["school"][0]
    val, (E, W) = ascent(G, m, 2.0, steps, seed)
    rho = irreps_of(G)
    th2 = (E ** 2).sum((0, 2)) + (W ** 2).sum(1); P2 = th2.sum()
    print(f"== {gname} m {m} seed {seed}: gamma/school {val/gs:.5f}; ||theta||^2 = {P2:.4f}")
    irr, classes = gc.programme_data(G)
    tot = {}
    for p in irr:
        nm = p["name"]; R = rho[nm]; d = R.shape[1]
        Es = np.array([[fhat(E[j, i], R) for i in range(m)] for j in range(2)]); Ws = np.array([fhat(W[i], R) for i in range(m)])
        X = sum(np.einsum("ab,cd,ef->abcdef", Es[0, i], Es[1, i], Ws[i]) for i in range(m))
        # invariant tensor for Phi(E1,E2,W) = tr(W E1^T E2^T) = sum W[k0,k1] E1[k2,k1] E2[k0,k2]
        T = np.zeros((d,) * 6)
        for k0 in range(d):
            for k1 in range(d):
                for k2 in range(d):
                    T[k2, k1, k0, k2, k0, k1] = 1
        Gc = (X * T).sum(); clean = Gc / d ** 3 * T; junk = np.linalg.norm(X - clean) / max(np.linalg.norm(X), 1e-30)
        mass = [(d / N) * (sum(np.linalg.norm(Es[j, i]) ** 2 for j in range(2)) + np.linalg.norm(Ws[i]) ** 2) for i in range(m)]
        tot[nm] = sum(mass)
        lam = [np.linalg.norm(Es[0, i]) * np.linalg.norm(Es[1, i]) * np.linalg.norm(Ws[i]) for i in range(m)]
        print(f"   {nm:9s} d {d}: G = {Gc:+.5f}, block junk fraction |X - proj_T X| / |X| = {junk:.4f}, weight mass {sum(mass):.4f} "
              f"({sum(mass)/P2:.3f} of total), active neurons {sum(1 for x in lam if x > 1e-3 * max(lam))}")
    # per-neuron irrep purity
    purity = []
    for i in range(m):
        e = np.array([tot and (irreps_mass(G, rho, E[:, i], W[i], nm)) for nm in rho])
        purity.append(e.max() / e.sum())
    print(f"   neuron irrep purity (energy share of top irrep): min {min(purity):.3f} median {np.median(purity):.3f}")

def irreps_mass(G, rho, Ei, Wi, nm):
    R = rho[nm]; d = R.shape[1]
    return (d / G.N) * (sum(np.linalg.norm(fhat(Ei[j], R)) ** 2 for j in range(Ei.shape[0])) + np.linalg.norm(fhat(Wi, R)) ** 2)

if __name__ == "__main__":
    anatomy("d3", 12, 0); anatomy("d7", 32, 0)
