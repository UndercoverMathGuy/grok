"""aball.py -- inradii of the junk-mode moment body of a support S (n = 2).

Modes of frequency k in Z_p^2:  m_{k,1} = (2k,0), m_{k,2} = (0,2k), m_{k,12} = (2k,2k).
Moment body  Omega = conv{ (exp(-2 pi i m.a/p))_m : a in Z_p^2 }  (moment vectors of probability densities on rows).
For a norm N on C^M:  inradius_N = min_{z != 0} h(z)/N(z),  h(z) = max_a Re sum_m z_m E[a,m]
                    = 1 / max{ N(z) : h(z) <= 1 }   (polar form, HOLES_aiming sec 1).
Any feasible z gives an UPPER bound on the inradius. Ascent: maximise the convex N over the polytope
{h <= 1} by iterating the LP  max Re<d, z>  with d = the sub-gradient direction of N at the current z.

Three quantities:
  Astar  : modes {(2k,0)}, N = l1        -- P3A's polydisc inradius A*(S) (calibration)
  Adisc  : modes 3K,       N = l1        -- polydisc inradius of the 3K body
  Aball  : modes 3K,       N = sum_k ||z^{(k)}||_2  -- l2-polyball inradius (what (ETA-A) needs)
"""
import argparse, json, sys, numpy as np
from scipy.optimize import linprog

def modes_of(S, which):
    S = list(S); M = []
    for k in S:
        if which == "K": M.append((2*k, 0))
        else: M += [(2*k, 0), (0, 2*k), (2*k, 2*k)]
    return np.array(M)

def char_matrix(p, M, torus=None):
    if torus is None:
        a = np.arange(p); A = np.stack(np.meshgrid(a, a, indexing="ij"), -1).reshape(-1, 2)
        return np.exp(-2j * np.pi * (A @ M.T) / p)
    # torus limit: rows are a fine grid of T^2, modes are real vectors (used only for K=1 scaling tests)
    g = np.linspace(0, 2*np.pi, torus, endpoint=False); A = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    return np.exp(-1j * (A @ M.T))

def ascend(E, groups, norm, rng, iters=60, tol=1e-10):
    """E: rows x M complex. groups: list of index arrays (per frequency). Returns best N(z) and z."""
    R, M = E.shape
    H = np.hstack([E.real, -E.imag])                     # Re(z E) = Re z * Re E - Im z * Im E
    bounds = [(-2, 2)] * (2*M)
    z = rng.standard_normal(M) + 1j * rng.standard_normal(M)
    best = -1
    for it in range(iters):
        if norm == 1: d = z / np.maximum(np.abs(z), 1e-15)
        else:
            d = np.zeros(M, complex)
            for g in groups:
                nz = np.linalg.norm(z[g]); d[g] = z[g] / nz if nz > 1e-15 else 0
        cobj = -np.concatenate([d.real, d.imag])           # maximise Re<d,z> = Re d.Re z + Im d.Im z
        res = linprog(cobj, A_ub=H, b_ub=np.ones(R), bounds=bounds, method="highs")
        if res.status != 0: break
        x = res.x; z = x[:M] + 1j * x[M:]
        val = np.abs(z).sum() if norm == 1 else sum(np.linalg.norm(z[g]) for g in groups)
        if val < best + tol: break
        best = val
    return best, z

def inradius(E, groups, norm, starts, seed=0):
    rng = np.random.default_rng(seed); top = 0; zb = None
    for s in range(starts):
        v, z = ascend(E, groups, norm, rng)
        if v > top: top, zb = v, z
    return 1 / top, zb

def run_support(p, S, starts):
    S = list(S); K = len(S)
    out = dict(p=p, S=S, K=K)
    E = char_matrix(p, modes_of(S, "K"))
    out["Astar"], _ = inradius(E, [np.arange(K)], 1, starts)
    E3 = char_matrix(p, modes_of(S, "3K")); groups = [np.arange(3*k, 3*k+3) for k in range(K)]
    out["Adisc"], _ = inradius(E3, groups, 1, starts)
    out["Aball"], zb = inradius(E3, groups, 2, starts)
    # anatomy of the worst direction for the polyball: per-frequency l2 mass and the pattern split
    w = np.array([np.linalg.norm(zb[g]) for g in groups]); w /= w.sum()
    pat = np.abs(zb.reshape(K, 3)); pat = (pat**2).sum(0); pat /= pat.sum()
    out["ball_dir_freq_share"] = np.round(w, 3).tolist(); out["ball_dir_pattern_share"] = np.round(pat, 3).tolist()
    print(f"p={p:4d} K={K} S={S}: A*={out['Astar']:.4f}  Adisc(3K)={out['Adisc']:.4f}  Aball(3K)={out['Aball']:.4f}"
          f"   ball worst dir: freq shares {out['ball_dir_freq_share']} pattern shares (1,2,12) {out['ball_dir_pattern_share']}", flush=True)
    return out

PAPERS = {59: [1,4,12,18,25,27], 71: [1,7,12,16,34], 97: [1,4,10,28,35,47], 113: [1,5,12,15,38,46], 127: [1,11,14,20,36,38,43]}
SMALL = {11: [1,3], 13: [1,2,5], 17: [1,3,5], 23: [1,2,5,7], 31: [1,3,7,12]}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int); ap.add_argument("--S", type=int, nargs="*")
    ap.add_argument("--papers", action="store_true"); ap.add_argument("--small", action="store_true")
    ap.add_argument("--torus", type=int, default=0, help="single-frequency torus limit, grid size")
    ap.add_argument("--starts", type=int, default=8); ap.add_argument("--out", default=None)
    a = ap.parse_args(); res = []
    if a.torus:
        M = np.array([(1, 0), (0, 1), (1, 1)], float); E = char_matrix(0, M, torus=a.torus); g = [np.arange(3)]
        print("torus K=1 3-mode body: disc inradius", inradius(E, g, 1, a.starts)[0], " ball inradius", inradius(E, g, 2, a.starts)[0])
        M1 = np.array([(1, 0)], float); print("torus K=1 1-mode: ", inradius(char_matrix(0, M1, torus=a.torus), [np.arange(1)], 1, 2)[0])
    if a.p and a.S: res.append(run_support(a.p, a.S, a.starts))
    if a.small:
        for p, S in SMALL.items(): res.append(run_support(p, S, a.starts))
    if a.papers:
        for p, S in PAPERS.items(): res.append(run_support(p, S, a.starts))
    if a.out: json.dump(res, open(a.out, "w"), indent=1)
