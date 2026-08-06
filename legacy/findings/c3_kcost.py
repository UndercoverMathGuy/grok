"""C3: do committees minimize weight cost sum a_k^(2/3) subject to the
margin-gap constraint, over menu subsets — with K free?

Key simplification: the constraint (sum_k a_k (1-cos(w_k x)) >= G* for all
x != 0) is linear in a, so cost(S; G*) = G*^(2/3) * cost(S; 1): the RANKING
of subsets and the argmin K are G*-independent. We set G* = 1.

Cost solve: minimize a concave objective over a polyhedron (min at a vertex);
we use SLSQP from 3 starts (scaled LP max-min allocation, equal allocation,
one perturbed) and keep the best. Approximation caveat noted in output.

Comparisons, to keep ourselves honest:
  - rank percentile of the CHOSEN committee among all menu-8 subsets with
    K in 3..6 (210 subsets), under the 2/3-power cost;
  - same under power 1 (= pure LP margin, the OLD functional that ranked
    chosen committees mediocrely) — C3's new content exists only if 2/3
    ranks chosen committees better than 1 does;
  - K prediction: argmin-K of the best subset vs observed K, vs the
    'always predict modal K=4' baseline;
  - blind top-K subset's rank (chosen==blind in loyal runs);
  - allocation efficiency: realized amplitude allocation (final coeffs,
    gap normalized to 1) vs optimal cost for the same S.
"""
import sys, time
from pathlib import Path
from itertools import combinations
import numpy as np
from scipy.optimize import minimize

ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "findings"))
from mask_lottery import discover, final_coeffs_and_acc, committee_from_coeffs
from margin_analysis import lp_relM, _cos_matrix

def cost_S(freqs, p, power):
    """min sum a^power s.t. sum_k a_k(1-cos(w_k x)) >= 1, a >= 0."""
    C = 1.0 - _cos_matrix(freqs, p)            # (p-1, K), all >= 0 rows
    K = C.shape[1]
    m, alloc = lp_relM(freqs, p)
    if m <= 1e-9:
        return np.inf, None                    # infeasible margin set
    starts = [alloc / m, np.ones(K) / (K * (C.mean() if False else 1))]
    # equal-alloc feasible scaling:
    eq = np.ones(K)
    g = (C @ eq).min()
    starts[1] = eq / g
    rng = np.random.default_rng(0)
    starts.append(starts[0] * (1 + 0.3 * rng.standard_normal(K)).clip(0.2))
    if power == 1.0:                           # exact LP answer
        return float(starts[0].sum()), starts[0]
    best, best_a = np.inf, None
    eps = 1e-9
    fun = lambda a: ((a + eps) ** power).sum()
    jac = lambda a: power * (a + eps) ** (power - 1.0)
    cons = [{"type": "ineq", "fun": lambda a: C @ a - 1.0,
             "jac": lambda a: C}]
    for s0 in starts:
        r = minimize(fun, np.maximum(s0, 0), jac=jac, method="SLSQP",
                     bounds=[(0, None)] * K, constraints=cons,
                     options={"maxiter": 300, "ftol": 1e-10})
        if r.success and ((C @ r.x).min() > 1 - 1e-6):
            if r.fun < best:
                best, best_a = r.fun, r.x
    return best, best_a

rows = []
t0 = time.time()
for d, cfg in discover():
    coeffs, acc, z = final_coeffs_and_acc(d, cfg)
    if acc < 0.99 or z is None:
        continue
    p = cfg.p
    final = committee_from_coeffs(coeffs)
    K = len(final)
    i3000 = int(np.argmin(np.abs(z["epochs"] - 3000)))
    menu = (np.argsort(np.abs(z["coeffs"][i3000]))[::-1][:8] + 1).tolist()
    blind = sorted(menu[:K])
    name = str(d.relative_to(ROOT / "runs"))
    if not set(final) <= set(menu):
        print(f"skip {name}: final not within menu-8")
        continue

    subs = [tuple(sorted(s)) for kk in (3, 4, 5, 6)
            for s in combinations(menu, kk)]
    costs23, costs1 = {}, {}
    for s in subs:
        costs23[s], _ = cost_S(list(s), p, 2/3)
        costs1[s], _ = cost_S(list(s), p, 1.0)
    fin = tuple(sorted(final)); bl = tuple(sorted(blind))
    order23 = sorted(subs, key=lambda s: costs23[s])
    order1 = sorted(subs, key=lambda s: costs1[s])
    r23 = order23.index(fin) + 1
    r1 = order1.index(fin) + 1
    rb23 = order23.index(bl) + 1
    kmin23 = len(order23[0])

    # allocation efficiency: realized amps on final committee
    amps = np.abs(coeffs)[np.array(final) - 1]
    C = 1.0 - _cos_matrix(final, p)
    gmin = (C @ amps).min()
    a_norm = amps / gmin                       # realized alloc scaled to gap 1
    real_cost = ((a_norm) ** (2/3)).sum()
    opt_cost = costs23[fin]
    rows.append(dict(name=name, K=K, kmin=kmin23, r23=r23, r1=r1, rb23=rb23,
                     n=len(subs), eff=opt_cost / real_cost,
                     reconf=fin != bl))
    print(f"{name:<34} K={K} argminK={kmin23} rank2/3={r23:>3}/{len(subs)} "
          f"rank1={r1:>3} blind2/3={rb23:>3} eff={opt_cost/real_cost:.3f}")

print(f"\n({time.time()-t0:.0f}s)")
import numpy as np
r23 = np.array([r["r23"] for r in rows]); n = np.array([r["n"] for r in rows])
r1 = np.array([r["r1"] for r in rows])
rb = np.array([r["rb23"] for r in rows])
pct23 = 100 * (r23 - 1) / (n - 1); pct1 = 100 * (r1 - 1) / (n - 1)
pctb = 100 * (rb - 1) / (n - 1)
print(f"chosen committee cost-rank percentile: 2/3-power mean {pct23.mean():.1f} "
      f"median {np.median(pct23):.1f} | power-1 mean {pct1.mean():.1f} "
      f"median {np.median(pct1):.1f}")
print(f"blind-draw percentile (2/3): mean {pctb.mean():.1f}")
from scipy.stats import wilcoxon
w = wilcoxon(pct23 - pct1)
print(f"paired 2/3 vs 1: mean diff {np.mean(pct23-pct1):+.1f}, wilcoxon p={w.pvalue:.4f}")
km = np.array([r["kmin"] for r in rows]); ko = np.array([r["K"] for r in rows])
print(f"K prediction: argmin-K == observed in {np.mean(km==ko)*100:.0f}% "
      f"(baseline always-4: {np.mean(ko==4)*100:.0f}%); argmin-K distribution "
      f"{np.bincount(km, minlength=7)[3:7]} vs observed {np.bincount(ko, minlength=7)[3:7]}")
eff = np.array([r["eff"] for r in rows])
print(f"allocation efficiency opt/realized (1 = realized is cost-optimal): "
      f"mean {eff.mean():.3f} min {eff.min():.3f} max {eff.max():.3f}")
# uniform-null for the mean percentile of chosen (n runs)
rng = np.random.default_rng(0)
nullm = np.array([rng.uniform(0, 100, len(pct23)).mean() for _ in range(50000)])
print(f"uniform-null one-sided p for chosen 2/3 percentile: "
      f"{(nullm <= pct23.mean()).mean():.5f}")
