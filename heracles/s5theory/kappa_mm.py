"""kappa_mm.py -- numerical upper bounds on the feature price kappa_{nu,r}(d) of a d-dim real-type irrep.

kappa(d) = N_r(T_d) / d^{nu r},  N_r(T) = inf { sum_i |mu_i|^r : T = sum_i mu_i U_i1 (x) ... (x) U_inu, ||U_ij||_F = 1 },
T_d = chain-trace (iterated matrix multiplication) tensor, <T_d, A_1 (x) ... (x) A_nu> = tr(A_1 ... A_nu), ||T_d||^2 = d^nu.
Variational form used here (AM-GM balances the factors at the optimum):
    N_r(T) = nu^{-q/2} min { sum_i ( sum_j ||A_ij||_F^2 )^{q/2} : sum_i A_i1 (x) ... (x) A_inu = T }.
Solved by an augmented Lagrangian (L-BFGS inner loop, torch fp64 autograd), many random starts and several term counts R.
Every value printed is the cost of an explicit decomposition whose residual is reported -> an UPPER bound on kappa
(up to the residual). Lower bounds are the theorems K2/K3 (flattening), printed alongside.
Usage: uv run python kappa_mm.py --nu 3 --q 2 --d 2 3 --starts 20
"""
import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

torch.set_default_dtype(torch.float64)
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"; OUT.mkdir(exist_ok=True)


def chain_tensor(d, nu):
    """T[a1,b1, a2,b2, ..., anu,bnu] = prod_j [b_j == a_{j+1}] (cyclic): <T, A_1 x ... x A_nu> = tr(A_1 ... A_nu)."""
    shape = [d, d] * nu
    T = np.zeros(shape)
    for idx in np.ndindex(*([d] * nu)):          # idx = (k_0, ..., k_{nu-1}); slot j = e_{k_j k_{j+1}}
        pos = []
        for j in range(nu):
            pos += [idx[j], idx[(j + 1) % nu]]
        T[tuple(pos)] = 1.0
    return torch.from_numpy(T.reshape([d * d] * nu))


def build(F, nu):
    """F (nu, R, d*d) -> sum_i F[0,i] x ... x F[nu-1,i] as a (d*d)^nu tensor."""
    letters = "abcdefgh"[:nu]
    spec = ",".join(f"z{l}" for l in letters) + "->" + letters
    return torch.einsum(spec, *[F[j] for j in range(nu)])


def cost(F, q):
    return ((F ** 2).sum((0, 2))) ** (q / 2)


def solve_one(T, d, nu, q, R, rng, init=None, iters_outer=30, rho0=10.0, verbose=False):
    D2 = d * d
    if init is None:
        scale = (d ** nu / R) ** (1 / nu) / math.sqrt(D2) * 1.0
        F0 = rng.normal(size=(nu, R, D2)) * scale
    else:
        F0 = init
    x = F0.reshape(-1).copy()
    lam = torch.zeros_like(T); rho = rho0
    Tn = float(T.norm())

    def fg(xv):
        F = torch.from_numpy(xv.reshape(nu, R, D2)).requires_grad_(True)
        Res = build(F, nu) - T
        L = cost(F, q).sum() + (lam * Res).sum() + 0.5 * rho * (Res ** 2).sum()
        g, = torch.autograd.grad(L, F)
        return float(L), g.numpy().reshape(-1).astype(np.float64)

    for it in range(iters_outer):
        r = minimize(fg, x, jac=True, method="L-BFGS-B", options=dict(maxiter=3000, gtol=1e-11, ftol=1e-15, maxcor=30))
        x = r.x
        with torch.no_grad():
            F = torch.from_numpy(x.reshape(nu, R, D2)); Res = build(F, nu) - T
            lam = lam + rho * Res
            res = float(Res.norm()) / Tn
        if verbose:
            print(f"   outer {it}: cost {float(cost(F, q).sum()) * nu ** (-q / 2):.6f} res {res:.2e} rho {rho:.1f}")
        if res < 1e-11:
            break
        rho = min(rho * 2, 1e7)
    F = torch.from_numpy(x.reshape(nu, R, D2))
    c = float(cost(F, q).sum()) * nu ** (-q / 2)
    # exact atom moduli (product of Frobenius norms) -> sum |mu|^r independent of balance
    mu = (F.norm(dim=2)).prod(0)
    Nr = float((mu ** (q / nu)).sum())
    return dict(Nr_balanced=c, Nr_atoms=Nr, res=res, R=R, n_active=int((mu > 1e-6 * mu.max()).sum()), mu=sorted(mu.tolist(), reverse=True)), x


def schoolbook(d, nu):
    """d^nu atoms e_{k0k1} x e_{k1k2} x ... ; returns F (nu, d^nu, d*d)."""
    atoms = list(np.ndindex(*([d] * nu)))
    F = np.zeros((nu, len(atoms), d * d))
    for i, idx in enumerate(atoms):
        for j in range(nu):
            F[j, i, idx[j] * d + idx[(j + 1) % nu]] = 1.0
    return F


def strassen():
    """Strassen's rank-7 decomposition of tr(ABC) for 2x2 (as A (x) B (x) C with C = transpose of the output map)."""
    # C_out = A B via Strassen; tr(A B C) = sum_{ij} (AB)_{ij} C_{ji}
    def E(i, j):
        M = np.zeros((2, 2)); M[i, j] = 1; return M
    A11, A12, A21, A22 = E(0, 0), E(0, 1), E(1, 0), E(1, 1)
    terms = [  # (A-coef, B-coef, output-coef matrix on (AB))
        (A11 + A22, A11 + A22, E(0, 0) + E(1, 1)),
        (A21 + A22, A11, E(1, 0) - E(1, 1)),
        (A11, A12 - A22, E(0, 1) + E(1, 1)),
        (A22, A21 - A11, E(0, 0) + E(1, 0)),
        (A11 + A12, A22, -E(0, 0) + E(0, 1)),
        (A21 - A11, A11 + A12, E(1, 1)),
        (A12 - A22, A21 + A22, E(0, 0)),
    ]
    F = np.zeros((3, 7, 4))
    for i, (a, b, c) in enumerate(terms):
        # sum_i <a, A> <b, B> (AB)_c ; (AB)_{kl} contributes to tr(ABC) with C_{lk}: C-factor = c^T
        F[0, i] = a.reshape(-1); F[1, i] = b.reshape(-1); F[2, i] = c.T.reshape(-1)
    return F


def lower_bounds(d, nu, r):
    lb = max(1.0, d ** ((nu - 1) * (1 - r))) if nu % 2 == 1 else d ** (nu * (1 - r))
    return lb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nu", type=int, default=3); ap.add_argument("--q", type=float, default=2.0)
    ap.add_argument("--d", type=int, nargs="+", default=[2]); ap.add_argument("--starts", type=int, default=10)
    ap.add_argument("--R", type=int, nargs="*", default=None); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    nu, q = a.nu, a.q; r = q / nu
    rng = np.random.default_rng(a.seed)
    results = {}
    for d in a.d:
        T = chain_tensor(d, nu)
        school = d ** nu; lb = lower_bounds(d, nu, r) * d ** (nu * r)
        print(f"\n== d {d} nu {nu} q {q:g} (r {r:.4f}): schoolbook N_r = {school}, kappa_school = {d ** (nu * (1 - r)):.4f}; proven lower kappa >= {lb / d ** (nu * r):.4f}", flush=True)
        Rs = a.R or sorted(set([max(1, d ** nu - 2), d ** nu, d ** nu + 2, int(1.5 * d ** nu)]))
        rows = []
        # checks: schoolbook and (d = 2) Strassen are exact decompositions
        Fs = schoolbook(d, nu)
        rs = float((build(torch.from_numpy(Fs), nu) - T).norm())
        print(f"   schoolbook residual {rs:.1e}", flush=True)
        if d == 2 and nu == 3:
            Fst = strassen(); rst = float((build(torch.from_numpy(Fst), nu) - T).norm())
            mu = np.prod(np.linalg.norm(Fst, axis=2), axis=0)
            print(f"   Strassen residual {rst:.1e}, raw sum mu^r {np.sum(mu ** r):.4f}", flush=True)
            info, _ = solve_one(T, d, nu, q, 7, rng, init=Fst + 1e-3 * rng.normal(size=Fst.shape))
            print(f"   Strassen polished (rank 7, balanced over the GL orbit by the solver): N_r {info['Nr_atoms']:.6f} res {info['res']:.1e}", flush=True)
            rows.append(dict(start="strassen", **info))
        info, _ = solve_one(T, d, nu, q, d ** nu, rng, init=Fs + 1e-2 * rng.normal(size=Fs.shape))
        print(f"   schoolbook + noise 1e-2 -> N_r {info['Nr_atoms']:.6f} res {info['res']:.1e} active {info['n_active']}", flush=True)
        rows.append(dict(start="school+noise", **info))
        for R in Rs:
            for s in range(a.starts):
                t0 = time.time()
                info, _ = solve_one(T, d, nu, q, R, rng)
                info["start"] = f"random R{R} s{s}"
                rows.append(info)
                print(f"   R {R:4d} start {s:2d}: N_r {info['Nr_atoms']:.6f} (balanced {info['Nr_balanced']:.6f}) res {info['res']:.1e} active {info['n_active']} "
                      f"kappa {info['Nr_atoms'] / d ** (nu * r):.5f} [{time.time() - t0:.0f}s]", flush=True)
        good = [x for x in rows if x["res"] < 1e-7]
        best = min(good, key=lambda x: x["Nr_atoms"]) if good else None
        results[d] = dict(school=school, lower=lb, best=best and {k: v for k, v in best.items() if k != "mu"},
                          best_mu=best and best["mu"][:40], all=[{k: v for k, v in x.items() if k != "mu"} for x in rows])
        if best:
            print(f"   BEST N_r {best['Nr_atoms']:.6f} = {best['Nr_atoms'] / school:.5f} x schoolbook; kappa <= {best['Nr_atoms'] / d ** (nu * r):.5f} "
                  f"(schoolbook {d ** (nu * (1 - r)):.5f}, lower {lb / d ** (nu * r):.5f}); active {best['n_active']}", flush=True)
    (OUT / f"kappa_nu{nu}_q{q:g}{a.tag}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
