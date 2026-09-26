"""analyse_sectors.py -- for D_n 2-dim irreps: decompose the (rho,rho,rho) block X of the junky ascent optima into the 8 parity
sectors (each factor conformal C+ = span{I, J} or anticonformal C- = span{R, RJ}); T lives on the 4 even sectors."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import gt_core as gc
from junk_ascent import ascent, ceilings
from verify_fourier import irreps_of, fhat
I = np.eye(2); J = np.array([[0., -1.], [1., 0.]]); R = np.array([[1., 0.], [0., -1.]]); RJ = R @ J
Cp = [I / np.sqrt(2), J / np.sqrt(2)]; Cm = [R / np.sqrt(2), RJ / np.sqrt(2)]
def proj(M, basis): return sum((M * B).sum() * B for B in basis)
def run(gname, m, seed):
    G = gc.get(gname); N = G.N; gs = ceilings(G, 3, 2.0)["school"][0]
    val, (E, W) = ascent(G, m, 2.0, 6000, seed)
    rho = irreps_of(G)
    print(f"== {gname} m {m} seed {seed}: gamma/school {val/gs:.5f}")
    for nm, Rr in rho.items():
        if Rr.shape[1] != 2: continue
        Es = np.array([[fhat(E[j, i], Rr) for i in range(m)] for j in range(2)]); Ws = np.array([fhat(W[i], Rr) for i in range(m)])
        X = sum(np.einsum("ab,cd,ef->abcdef", Es[0, i], Es[1, i], Ws[i]) for i in range(m))
        tot = (X ** 2).sum(); rep = []
        for s1 in "+-":
            for s2 in "+-":
                for s3 in "+-":
                    B = [Cp if s == "+" else Cm for s in (s1, s2, s3)]
                    Xs = sum((X * np.einsum("ab,cd,ef->abcdef", b1, b2, b3)).sum() * np.einsum("ab,cd,ef->abcdef", b1, b2, b3) for b1 in B[0] for b2 in B[1] for b3 in B[2])
                    rep.append(f"{s1}{s2}{s3}:{(Xs**2).sum()/tot:.3f}")
        # per-neuron: which factors are conformal / anticonformal
        kinds = []
        for i in range(m):
            lam = np.linalg.norm(Es[0, i]) * np.linalg.norm(Es[1, i]) * np.linalg.norm(Ws[i])
            if lam < 1e-3 * max(1e-12, max(np.linalg.norm(Es[0, k]) * np.linalg.norm(Es[1, k]) * np.linalg.norm(Ws[k]) for k in range(m))): continue
            k = "".join("+" if np.linalg.norm(proj(Mx, Cp)) > 0.99 * np.linalg.norm(Mx) else ("-" if np.linalg.norm(proj(Mx, Cm)) > 0.99 * np.linalg.norm(Mx) else "?") for Mx in (Es[0, i], Es[1, i], Ws[i]))
            kinds.append(k)
        print(f"   {nm}: sector energy shares {' '.join(rep)}; active-neuron factor types {kinds}")
if __name__ == "__main__":
    run("d3", 12, 0); run("d7", 32, 0)
