"""P3C step 4: end-to-end fp64 verification of THEOREM P3C-1 and of §2.2's witness.

Three independent checks, all on explicit objects, nothing optimised:

  --block ident   the profile identity  Lambda_D(a) = sum_k Re[W_kD h_k(u^(k)(a))]
                  against the brute-force p^n x p logit table of a pure network.
  --block mm      the M_m family of P3C.md §2.2: closed-form C_T, eta, cost, rho.
  --block chain   THEOREM P3C-1 on explicit random pure networks: build the
                  partition density w as an actual non-negative table on Z_p^n by
                  solving for each block's nu_i with an LP, check w >= 0 and
                  sum w = 1, then check  2^n * margin <= sum_k w_kD Re g_P^(k)
                  for the claimed D and phases, and finally that the whole chain
                  of THEOREM P3C-1 lands on the target.

  uv run python p3c_verify.py --block ident mm chain
"""
import argparse, json, math, os
import numpy as np
from scipy.optimize import linprog

import p3c_core as C
import p3c_part as PP


# ------------------------------------------------------------------ the network
def rand_pure_net(p, n, S, rng, Nk=None):
    """A pure single-frequency chain network.  Returns per-frequency atom data
    (c, V) and the raw weights (E_j, W) so the logits can be built by brute force."""
    neurons = []
    pops = {}
    for k in S:
        N = Nk if Nk else int(rng.integers(1, 4))
        alpha = rng.uniform(0, 2 * np.pi, (N, n))
        beta = rng.uniform(0, 2 * np.pi, N)
        rr = rng.uniform(0.5, 1.5, (N, n))
        rho = rng.uniform(0.5, 1.5, N)
        c = (rr.prod(axis=1) * rho) * np.exp(1j * (alpha.sum(axis=1) - beta))
        V = np.exp(-2j * alpha)
        pops[k] = (c, V)
        for i in range(N):
            neurons.append((k, rr[i], alpha[i], rho[i], beta[i]))
    return pops, neurons


def logits(p, n, neurons):
    """z_c(x) for every row x in Z_p^n and every class c: shape (p^n, p)."""
    rows = np.array(np.meshgrid(*[np.arange(p)] * n, indexing="ij")).reshape(n, -1).T
    z = np.zeros((rows.shape[0], p))
    cls = np.arange(p)
    for (k, rr, alpha, rho, beta) in neurons:
        act = np.ones(rows.shape[0])
        for j in range(n):
            act = act * (rr[j] * np.cos(2 * np.pi * k * rows[:, j] / p + alpha[j]))
        wout = rho * np.cos(2 * np.pi * k * cls / p + beta)
        z += np.outer(act, wout)
    return rows, z


def exact_margin(p, n, neurons):
    rows, z = logits(p, n, neurons)
    corr = (rows.sum(axis=1)) % p
    zc = z[np.arange(z.shape[0]), corr]
    gap = zc[:, None] - z
    gap[np.arange(z.shape[0]), corr] = np.inf
    return float(gap.min()), rows, z, corr


def lambda_profile(p, n, pops, rows):
    """sum_k Re[W_kD h_k(u^(k)(a))] for every row a and every D != 0."""
    w = np.exp(2j * np.pi / p)
    out = np.zeros((rows.shape[0], p))
    for k, (c, V) in pops.items():
        Ct = C.coeffs(c, V)                          # over bitmasks
        n_ = V.shape[1]
        sig = np.zeros((rows.shape[0], 1 << n_), dtype=int)
        for t in range(1 << n_):
            s = np.zeros(rows.shape[0], dtype=int)
            for j in range(n_):
                if (t >> j) & 1:
                    s = s + rows[:, j]
            sig[:, t] = (2 * k * s) % p
        h = (np.power(w, -sig) * Ct[None, :]).sum(axis=1)     # (rows,)
        D = np.arange(p)
        W = 1 - np.power(w, (-k * D) % p)
        out += np.real(h[:, None] * W[None, :])
    return out                                        # column D = 0 is zero


# ---------------------------------------------------------------- the densities
def block_density(p, S, A, phases):
    """A probability density nu on Z_p with nuhat(2k) = A e^{i phases[k]} for k in S.
    Returns None if the LP is infeasible (i.e. A exceeds the inradius in that
    direction)."""
    w = np.exp(2j * np.pi / p)
    rowsA, rhs = [], []
    for idx, k in enumerate(S):
        e = np.power(w, (-2 * k * np.arange(p)) % p)
        tgt = A * np.exp(1j * phases[idx])
        rowsA.append(e.real); rhs.append(tgt.real)
        rowsA.append(e.imag); rhs.append(tgt.imag)
    rowsA.append(np.ones(p)); rhs.append(1.0)
    res = linprog(np.zeros(p), A_eq=np.array(rowsA), b_eq=np.array(rhs),
                  bounds=[(0, None)] * p, method="highs")
    return res.x if res.success else None


def astar_dir(p, S, phases, lo=0.0, hi=1.0, it=60):
    """Largest A realisable in THIS phase direction (bisection on the LP)."""
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if block_density(p, S, mid, phases) is None:
            hi = mid
        else:
            lo = mid
    return lo


def astar(p, S, rng, trials=4000):
    """A*(p, {2k}) = min over unit z of [max_a Re sum_i z_i omega^{-2k_i a}]/sum|z_i|,
    by random search + local polish (an UPPER bound on the true A*)."""
    w = np.exp(2j * np.pi / p)
    E = np.array([np.power(w, (-2 * k * np.arange(p)) % p) for k in S])   # (K,p)
    best = np.inf
    for _ in range(trials):
        z = rng.standard_normal(len(S)) + 1j * rng.standard_normal(len(S))
        val = np.max(np.real(z @ E)) / np.sum(np.abs(z))
        best = min(best, val)
    return float(best)


# --------------------------------------------------------------------- blocks
def block_ident(out, rng):
    worst = 0.0
    for p, n, S in [(7, 2, [1, 2]), (11, 2, [1, 4]), (11, 3, [1, 3]),
                    (13, 2, [2, 3, 6]), (13, 3, [1, 5])]:
        pops, neurons = rand_pure_net(p, n, S, rng)
        m_exact, rows, z, corr = exact_margin(p, n, neurons)
        L = lambda_profile(p, n, pops, rows)
        # Lambda_D(a) should equal 2^n (z_correct - z_{correct+D})
        gap = np.zeros_like(L)
        for D in range(1, p):
            gap[:, D] = z[np.arange(z.shape[0]), corr] - \
                z[np.arange(z.shape[0]), (corr + D) % p]
        err = np.abs(L[:, 1:] - 2 ** n * gap[:, 1:]).max() / \
            max(1e-30, np.abs(2 ** n * gap[:, 1:]).max())
        lam_min = L[:, 1:].min()
        err2 = abs(lam_min - 2 ** n * m_exact) / max(1e-30, abs(2 ** n * m_exact))
        worst = max(worst, err, err2)
        print(f"  p={p} n={n} S={S}: rel err of the identity {err:.3e}, "
              f"of 2^n*margin {err2:.3e}", flush=True)
    out["ident_worst_rel_err"] = worst
    print(f"  WORST relative error {worst:.3e}")


def block_mm(out, rng):
    rows = []
    for nu, q in [(3, 2.0), (3, 2.5), (4, 2.0), (4, 3.0), (5, 2.0)]:
        n, r = nu - 1, q / nu
        k0, rn = C.kappa0(n, r), C.rho_need(n, r)
        for m in range(1, n + 2):
            c = np.full(m, 1.0 / m, dtype=complex)
            V = np.exp(1j * np.array([[-2 * np.pi * l / m] * n for l in range(m)]))
            clean, eta, cost = C.report(c, V, r)
            # closed forms
            eta_cf = np.sqrt(sum(
                math.comb(n, j) for j in range(1, n + 1) if j % m == 0))
            cost_cf = m ** (1 - r)
            rho = np.inf if eta < 1e-12 else (1 - (cost ** (1 / r)) / k0) / eta
            rows.append({"nu": nu, "q": q, "m": m, "eta": eta, "eta_cf": float(eta_cf),
                         "cost": cost, "cost_cf": cost_cf,
                         "A": cost ** (1 / r), "rho": float(rho),
                         "rho_need": rn,
                         "refutes": bool(eta > 1e-12 and rho > rn)})
            print(f"  nu={nu} q={q} m={m}: clean={clean.real:.12f}{clean.imag:+.1e}i "
                  f"eta={eta:.12f} (cf {eta_cf:.12f})  cost={cost:.12f} "
                  f"(cf {cost_cf:.12f})  A={cost**(1/r):.9f}  rho={rho:.9f} "
                  f"vs rho_need={rn:.9f}  {'REFUTES' if (eta > 1e-12 and rho > rn) else ''}", flush=True)
    out["mm"] = rows


def block_chain(out, rng, ntest=8):
    recs = []
    for p, n, S in [(11, 2, [1, 4]), (13, 2, [2, 3, 6]), (11, 3, [1, 4]),
                    (17, 2, [2, 5, 8])]:
        As = astar(p, S, rng)
        parts = PP.partitions(n)
        for t in range(ntest):
            pops, neurons = rand_pure_net(p, n, S, rng)
            m_exact, rows, z, corr = exact_margin(p, n, neurons)
            # pick the best partition by the aggregate criterion at q = 2
            r = 2.0 / (n + 1)
            costs = np.array([float(np.sum(np.abs(pops[k][0]) ** r)) for k in S])
            Rmat = np.array([PP.R_all(pops[k][0], pops[k][1], As, parts, r, n,
                                      restarts=12, sweeps=50, rng=rng) for k in S])
            b = costs / costs.sum()
            agg = (b[:, None] * Rmat ** r).sum(axis=0)
            iP = int(np.argmin(agg))
            blocks = parts[iP]
            # realise the density EXPLICITLY and check the bound it gives
            ok = True
            phis = []
            for k in S:
                Ct = C.coeffs(*pops[k])
                Dc = PP.part_coef(Ct, blocks, As)
                _, phi = PP.min_re(Dc, len(blocks), restarts=16, sweeps=60, rng=rng)
                phis.append(phi)                     # (K, s)
            phis = np.array(phis)
            dens = []
            for i in range(len(blocks)):
                nu_i = block_density(p, S, As * (1 - 1e-9), phis[:, i])
                if nu_i is None:
                    ok = False
                    break
                dens.append(nu_i)
            if not ok:
                recs.append({"p": p, "n": n, "S": S, "lp": "infeasible"})
                continue
            # explicit density on Z_p^n
            w = np.zeros(p ** n)
            idx = rows
            wv = np.ones(p ** n)
            for i, bi in enumerate(blocks):
                js = [j for j in range(n) if (bi >> j) & 1]
                sig = idx[:, js].sum(axis=1) % p
                wv = wv * dens[i][sig] / p ** (len(js) - 1)
            w = wv
            assert w.min() >= -1e-12, w.min()
            # E_w[Lambda_D] for every D, and the theorem's bound
            L = lambda_profile(p, n, pops, rows)
            EwL = w @ L
            both = np.array([min(EwL[D], EwL[p - D]) for D in range(1, p)])
            bound_pm = float(both.min())
            lam_true = 2 ** n * m_exact
            # the full chain
            kap = C.kappa0(n, r)
            y = np.maximum(Rmat[:, iP], 0) * costs ** (1 / r) / kap
            Gam = gamma_prog(p, 1.0 / r, S)
            chain = Gam * (np.sum(y ** r)) ** (1 / r)
            target = Gam * kap ** -1 * (costs.sum()) ** (1 / r)
            recs.append({"p": p, "n": n, "S": S, "A": As, "partition": list(blocks),
                         "w_min": float(w.min()), "w_sum": float(w.sum()),
                         "Lambda": lam_true, "EwL_pm_min": float(bound_pm),
                         "chain_bound": float(chain), "target": float(target),
                         "agg_criterion": float(agg[iP]),
                         "ok_step3": bool(lam_true <= bound_pm + 1e-9),
                         "ok_chain": bool(lam_true <= chain + 1e-9),
                         "gamma_ratio": float(lam_true / target)})
            print(f"  p={p} n={n} S={S} P={blocks} A*={As:.6f}: "
                  f"Lambda={lam_true:.9f} <= E_w[+-D] {bound_pm:.9f} "
                  f"<= chain {chain:.9f}; target {target:.9f}; "
                  f"agg={agg[iP]:.6f}  gamma/target={lam_true/target:.6f}", flush=True)
    out["chain"] = recs


def gamma_prog(p, s, S=None):
    """Gamma_s(p) = max_{b>=0,sum b=1} min_{D!=0} sum_k b_k^s (1-cos 2 pi k D/p),
    over the FULL frequency set (an upper bound for any sub-support)."""
    from scipy.optimize import minimize
    ks = np.arange(1, (p - 1) // 2 + 1)
    Ds = np.arange(1, p)
    Wm = 1 - np.cos(2 * np.pi * np.outer(ks, Ds) / p)

    def neg(u):
        b = np.exp(u); b = b / b.sum()
        return -np.min((b ** s) @ Wm)
    best = 0.0
    rng = np.random.default_rng(5)
    for _ in range(40):
        u0 = rng.standard_normal(len(ks))
        res = minimize(neg, u0, method="Nelder-Mead",
                       options={"maxiter": 20000, "fatol": 1e-14, "xatol": 1e-12})
        best = max(best, -res.fun)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", nargs="*", default=["ident", "mm", "chain"])
    ap.add_argument("--out", default="FINAL_RUNS/P3C/verify.json")
    a = ap.parse_args()
    rng = np.random.default_rng(31415)
    out = {}
    if "ident" in a.block:
        print("== profile identity"); block_ident(out, rng)
    if "mm" in a.block:
        print("== the M_m family"); block_mm(out, rng)
    if "chain" in a.block:
        print("== THEOREM P3C-1 end to end"); block_chain(out, rng)
    if "tight" in a.block:
        print("== THEOREM P3C-1 at the equality case"); block_tight(out, rng)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print("wrote", a.out)





# ------------------------------------------------- the tight (near-optimal) test
def block_tight(out, rng, ntrial=12):
    """The meaningful end-to-end test: start from the SHARE-WEIGHTED (n+1)-block
    on the Gamma-optimal support -- which attains the target exactly -- then
    inject junk (swap a frequency's block for an M_m, or perturb it), and check

        2^n margin  <=  E_w[Lambda_{+-D}]  <=  chain bound  <=  target ,

    with w an explicitly constructed non-negative density on Z_p^n.
    """
    import p3c_astar as AS
    recs = []
    for p, n, S in [(11, 2, [1, 4]), (13, 2, [2, 3, 6]), (17, 2, [2, 5, 8])]:
        r = 2.0 / (n + 1)
        s = 1.0 / r
        Gam, bopt = AS.gamma_of(p, S, s, tries=20)
        A, _ = AS.astar_upper(p, S, rng, trials=20000, polish=10)
        parts = PP.partitions(n)
        kap = C.kappa0(n, r)
        nb = n + 1
        w0 = np.exp(2j * np.pi / nb)
        for trial in range(ntrial):
            x = bopt ** (1.0 / r)
            pops, neurons = {}, []
            for i, k in enumerate(S):
                if trial > 0 and i == trial % len(S):
                    m = 1 + (trial % n) if n > 1 else 1      # an M_m frequency
                    cc = np.full(m, x[i] / m, dtype=complex)
                    th = np.array([[-2 * np.pi * l / m] * n for l in range(m)])
                else:
                    cc = np.full(nb, x[i] / nb, dtype=complex)
                    th = np.array([[-2 * np.pi * l / nb] * n for l in range(nb)])
                if trial > 1:
                    th = th + 0.15 * rng.standard_normal(th.shape)
                    cc = cc * (1 + 0.1 * rng.standard_normal(len(cc)))
                pops[k] = (cc, np.exp(1j * th))
                # realise as neurons: alpha_j = -th/2, beta = sum alpha - arg c
                al = -np.angle(np.exp(1j * th)) / 2.0
                be = al.sum(axis=1) - np.angle(cc)
                rr = np.ones_like(al)
                rho = np.abs(cc)
                for l in range(len(cc)):
                    neurons.append((k, rr[l], al[l], rho[l], be[l]))
            m_exact, rows, z, corr = exact_margin(p, n, neurons)
            costs = np.array([float(np.sum(np.abs(pops[k][0]) ** r)) for k in S])
            Rmat = np.array([PP.R_all(pops[k][0], pops[k][1], A, parts, r, n,
                                      restarts=16, sweeps=60, rng=rng) for k in S])
            b = costs / costs.sum()
            agg = (b[:, None] * Rmat ** r).sum(axis=0)
            iP = int(np.argmin(agg)); blocks = parts[iP]
            phis = []
            for k in S:
                Dc = PP.part_coef(C.coeffs(*pops[k]), blocks, A)
                _, phi = PP.min_re(Dc, len(blocks), restarts=24, sweeps=80, rng=rng)
                phis.append(phi)
            phis = np.array(phis)
            dens, ok = [], True
            for i in range(len(blocks)):
                nu_i = block_density(p, S, A * (1 - 1e-9), phis[:, i])
                if nu_i is None:
                    ok = False; break
                dens.append(nu_i)
            if not ok:
                print("  LP infeasible (A above the inradius in this direction)")
                continue
            wv = np.ones(p ** n)
            for i, bi in enumerate(blocks):
                js = [j for j in range(n) if (bi >> j) & 1]
                sig = rows[:, js].sum(axis=1) % p
                wv = wv * dens[i][sig] / p ** (len(js) - 1)
            L = lambda_profile(p, n, pops, rows)
            EwL = wv @ L
            bound_pm = float(min(min(EwL[D], EwL[p - D]) for D in range(1, p)))
            lam_true = 2 ** n * m_exact
            y = np.maximum(Rmat[:, iP], 0) * costs ** (1 / r) / kap
            chain = Gam * (np.sum(y ** r)) ** (1 / r)
            target = Gam * kap ** -1 * (costs.sum()) ** (1 / r)
            recs.append({"p": p, "n": n, "S": S, "trial": trial, "A": A,
                         "partition": list(blocks), "w_min": float(wv.min()),
                         "w_sum": float(wv.sum()), "Lambda": lam_true,
                         "EwL_pm": bound_pm, "chain": float(chain),
                         "target": float(target), "agg": float(agg[iP]),
                         "ratio_to_target": float(lam_true / target),
                         "ok1": bool(lam_true <= bound_pm + 1e-9),
                         "ok2": bool(bound_pm <= chain + 1e-9),
                         "ok3": bool(chain <= target + 1e-9)})
            print(f"  p={p} t={trial} P={blocks}: Lambda={lam_true:.9f} "
                  f"<= Ew{bound_pm:.9f} <= chain {chain:.9f} <= target {target:.9f} "
                  f"| gamma/target={lam_true/target:.9f} agg={agg[iP]:.6f} "
                  f"w>=0:{wv.min() >= -1e-12} "
                  f"{'OK' if (lam_true <= bound_pm + 1e-9 and bound_pm <= chain + 1e-9 and chain <= target + 1e-9) else 'VIOLATION'}",
                  flush=True)
    out["tight"] = recs
    bad = [x for x in recs if not (x["ok1"] and x["ok2"] and x["ok3"])]
    print(f"  violations: {len(bad)} of {len(recs)}")


if __name__ == "__main__":
    main()
