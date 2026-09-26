"""P3C step 6: A*(p, {2k : k in S}) -- measured, and a RIGOROUS lower bound.

    A*  =  min over z in C^K, z != 0, of  [ max_a Re sum_i z_i omega^{-2 k_i a} ] / sum_i |z_i|

("the polydisc inradius of the moment body": the largest A such that for EVERY
phase vector there is a probability density nu on Z_p with nuhat(2k) = A e^{i phi_k}
for all k in S").  Plain reading: how deep a non-negative test density can aim at
all K frequencies at once with phases of my choosing.

The minimum is non-convex, so an optimiser only gives an UPPER bound -- which is
the wrong side for THEOREM P3C-1.  A rigorous LOWER bound comes for free from
Lipschitz continuity: writing F(z) = max_a Re sum_i z_i e_a(i) with |e_a(i)| = 1,

    |F(z) - F(z')|  <=  sum_i |z_i - z'_i|   =   ||z - z'||_1 ,

so F is 1-Lipschitz for the l_1 norm.  Minimising over the l_1 unit sphere of
C^K (real dimension 2K-1) on a mesh of l_1-radius `delta` therefore gives

    A*  >=  min over the mesh of F  -  delta .

  uv run python p3c_astar.py --p 11 13 17 19 23 --mesh 0.02
"""
import argparse, itertools, json, os
import numpy as np
from scipy.optimize import minimize


def Emat(p, S):
    w = np.exp(2j * np.pi / p)
    return np.array([np.power(w, (-2 * k * np.arange(p)) % p) for k in S])  # (K,p)


def F(z, E):
    return float(np.max(np.real(z @ E)))


def astar_upper(p, S, rng, trials=20000, polish=40):
    E = Emat(p, S)
    K = len(S)
    best, bz = np.inf, None
    for _ in range(trials):
        z = rng.standard_normal(K) + 1j * rng.standard_normal(K)
        z = z / np.sum(np.abs(z))
        v = F(z, E)
        if v < best:
            best, bz = v, z
    for _ in range(polish):
        x0 = np.concatenate([bz.real, bz.imag]) + 0.05 * rng.standard_normal(2 * K)

        def obj(x):
            z = x[:K] + 1j * x[K:]
            s = np.sum(np.abs(z))
            return F(z, E) / s if s > 1e-12 else 1e9
        res = minimize(obj, x0, method="Nelder-Mead",
                       options={"maxiter": 20000, "xatol": 1e-12, "fatol": 1e-14})
        if res.fun < best:
            best = float(res.fun)
            z = res.x[:K] + 1j * res.x[K:]
            bz = z / np.sum(np.abs(z))
    return best, bz


def astar_lower(p, S, mesh):
    """Rigorous lower bound by an l_1-sphere mesh + the 1-Lipschitz property.

    Parametrise z_i = s_i e^{i t_i} with s >= 0, sum s = 1.  A mesh of step h on
    the simplex (in l_1) and of step h on each phase (which moves z by at most
    s_i * h <= h in l_1) gives a covering of l_1-radius <= 2h... we take the
    conservative delta = 2h."""
    E = Emat(p, S)
    K = len(S)
    h = mesh
    ns = max(2, int(np.ceil(1.0 / h)))
    nt = max(4, int(np.ceil(2 * np.pi / h)))
    # simplex grid with step 1/ns
    best = np.inf
    ts = np.arange(nt) * 2 * np.pi / nt
    # by the global phase gauge (z -> e^{i c} z leaves F invariant? NO -- F uses Re,
    # so no gauge).  Enumerate all phases.
    grids = list(itertools.product(range(nt), repeat=K))
    phase = np.exp(1j * ts)
    for comp in _simplex(ns, K):
        s = np.array(comp) / ns
        for g in grids:
            z = s * phase[list(g)]
            v = F(z, E)
            if v < best:
                best = v
    delta = h + h          # simplex step (l_1) + phase step (l_1 <= sum s_i h = h)
    return best - delta, best, delta, ns, nt


def _simplex(ns, K):
    if K == 1:
        yield (ns,)
        return
    for i in range(ns + 1):
        for rest in _simplex(ns - i, K - 1):
            yield (i,) + rest


def gamma_of(p, S, s, tries=12):
    """Gamma_s(p,S) = max_{b on S} min_{D != 0} sum_k b_k^s (1 - cos 2 pi k D/p),
    and the number of shares it actually uses."""
    from scipy.optimize import minimize as mz
    Ds = np.arange(1, p)
    Wm = 1 - np.cos(2 * np.pi * np.outer(list(S), Ds) / p)

    def neg(u):
        b = np.exp(u); b = b / b.sum()
        return -np.min((b ** s) @ Wm)
    best, bb = -np.inf, None
    for seed in range(tries):
        rng = np.random.default_rng(seed)
        res = mz(neg, rng.standard_normal(len(S)), method="Nelder-Mead",
                 options={"maxiter": 20000, "fatol": 1e-14, "xatol": 1e-12})
        if -res.fun > best:
            b = np.exp(res.x); b = b / b.sum()
            best, bb = -float(res.fun), b
    return best, bb


def gamma_support_old(p, s=1.5):
    """The Gamma_s-optimal support at this p (brute force over supports is done
    upstream in kstar.py; here we just take the argmax over subsets of size <= 5
    of min_D sum b_k^s w_kD at the optimal b, by a short optimisation)."""
    from scipy.optimize import minimize as mz
    ks = list(range(1, (p - 1) // 2 + 1))
    Ds = np.arange(1, p)
    best = (-1, None)
    for K in range(1, min(6, len(ks)) + 1):
        for Ssub in itertools.combinations(ks, K):
            Wm = 1 - np.cos(2 * np.pi * np.outer(Ssub, Ds) / p)

            def neg(u):
                b = np.exp(u); b = b / b.sum()
                return -np.min((b ** s) @ Wm)
            v = -np.inf
            for seed in range(6):
                rng = np.random.default_rng(seed)
                res = mz(neg, rng.standard_normal(K), method="Nelder-Mead",
                         options={"maxiter": 8000, "fatol": 1e-13})
                v = max(v, -res.fun)
            if v > best[0]:
                best = (v, Ssub)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, nargs="*", default=[11, 13, 17, 19, 23])
    ap.add_argument("--S", type=int, nargs="*", default=None)
    ap.add_argument("--mesh", type=float, default=0.03)
    ap.add_argument("--nu", type=int, default=3)
    ap.add_argument("--q", type=float, default=2.0)
    ap.add_argument("--out", default="FINAL_RUNS/P3C/astar.json")
    a = ap.parse_args()
    s = a.nu / a.q
    rng = np.random.default_rng(2718)
    out = []
    for p in a.p:
        if a.S:
            S, gam = list(a.S), None
        else:
            gam, S = gamma_support(p, s)
            S = list(S)
        up, z = astar_upper(p, S, rng)
        lo, mn, delta, ns, nt = astar_lower(p, S, a.mesh) if len(S) <= 2 \
            else (None, None, None, None, None)
        out.append({"p": p, "S": S, "Gamma_s": gam, "s": s,
                    "astar_upper": up,
                    "astar_lower_rigorous": lo, "mesh_min": mn, "mesh_delta": delta})
        print(f"p={p} S={S}: A* measured (upper) {up:.9f}; rigorous lower "
              f"{'n/a' if lo is None else f'{lo:.9f}'} "
              f"(mesh min {mn if mn is None else round(mn,9)}, delta {delta})",
              flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print("wrote", a.out)





def scan_supports(p, s, rng, out):
    """Every frequency support at this p: A* of its mode set, Gamma_s(p,S), and
    the 'pressure' Gamma_s(p,S)/Gamma_s(p) that THEOREM P3C-1's budget relaxes by."""
    ks = list(range(1, (p - 1) // 2 + 1))
    Gfull, _ = gamma_of(p, ks, s)
    rows = []
    for K in range(1, len(ks) + 1):
        for S in itertools.combinations(ks, K):
            g, b = gamma_of(p, list(S), s, tries=8)
            A, _ = astar_upper(p, list(S), rng, trials=4000, polish=8)
            rows.append({"p": p, "S": list(S), "K": K, "gamma": g,
                         "press": g / Gfull, "astar": A,
                         "n_eff": int(np.sum(b > 1e-4))})
    rows.sort(key=lambda x: -x["press"])
    out.append({"p": p, "Gamma_full": Gfull, "rows": rows})
    print(f"p={p}: Gamma_s = {Gfull:.9f}; {len(rows)} supports", flush=True)
    for x in rows[:6] + rows[-3:]:
        print(f"   S={x['S']} K={x['K']} (eff {x['n_eff']}) "
              f"gamma/Gamma={x['press']:.6f}  A*={x['astar']:.6f}", flush=True)


def main_scan():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, nargs="*", default=[11, 13])
    ap.add_argument("--nu", type=int, default=3)
    ap.add_argument("--q", type=float, default=2.0)
    ap.add_argument("--out", default="FINAL_RUNS/P3C/astar_scan.json")
    a = ap.parse_args()
    rng = np.random.default_rng(2718)
    out = []
    for p in a.p:
        scan_supports(p, a.nu / a.q, rng, out)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print("wrote", a.out)





def main_papers():
    """A* of the Gamma-optimal supports at the paper's primes (from kstar.json)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="nu3_q2")
    ap.add_argument("--out", default="FINAL_RUNS/P3C/astar_papers.json")
    a = ap.parse_args()
    d = json.load(open("kstar.json"))
    rng = np.random.default_rng(11)
    rows = []
    for ps, v in d.items():
        p = int(ps)
        Ks = v["pred"][a.cell]["K_star"]
        for K in sorted({Ks - 1, Ks, Ks + 1}):
            S = v["S"].get(str(K))
            if not S:
                continue
            A, _ = astar_upper(p, S, rng, trials=30000, polish=15)
            rows.append({"p": p, "K": K, "K_star": Ks, "S": S, "astar": A})
            print(f"p={p} K={K} (K*={Ks}) S={S}: A* = {A:.6f}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1, default=float)
    print("wrote", a.out)


if __name__ == "__main__":
    import sys
    if "--scan" in sys.argv:
        sys.argv.remove("--scan")
        main_scan()
    elif "--papers" in sys.argv:
        sys.argv.remove("--papers")
        main_papers()
    else:
        main()
