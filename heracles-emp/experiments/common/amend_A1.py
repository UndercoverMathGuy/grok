"""Amendment A1 (POST HOC, not preregistered): re-scoring of E1/E1b, E6, E7, E8 with
  (i)   energy-share counts, the definitions of the earlier programme (heracles/nsum.py `analyse`, reimplemented here):
        e_j = A_j^2 / sum_k A_k^2 over the clean amplitudes A; K_2pct = #{e_j >= 0.02}, K_5pct = #{e_j >= 0.05};
        N_05 / N_10 = fewest frequencies carrying 95% / 90% of the clean energy;
  (ii)  programme efficiency eff = programme_value(A, p, s) / Gamma_s(p): the programme's objective at the measured
        allocation, as a fraction of the optimum. It is 1 exactly when the allocation is an optimal solution (any
        optimal vertex, any dilate), so it does not penalise landing on a near-tied optimum instead of y*;
  (iii) eps-optimal vertex sets (vertex-cache primes p <= 37 only): V_eps = {vertices v : value(v) >= (1-eps) Gamma};
        [K_min, K_max] of V_eps, and share TV / cosine to the best-aligned v in V_eps (registered alignment rule:
        max cosine) plus the minimum TV over V_eps.
Reads pulled runs only (light; runs on the laptop in seconds). Writes experiments/EXP/analysis/A1_per_run.csv and
A1_summary.json for EXP in E1, E6, E7, E8, and experiments/common/A1_summary.json.
Usage: uv run python experiments/common/amend_A1.py
"""
import json, os, sys, time
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np
import alib
from core import metrics as MT, measure as MS, programme as PG

EPS = (0.01,)  # eps = 0.05 dropped: at p = 37, s ~ 1.2 the set holds up to 62k orbits (~1M vertices), so matching it is vacuous
T0 = time.time()


def energy_counts(A):
    A = np.clip(np.asarray(A, float), 0, None)
    e = A ** 2
    if e.sum() <= 0:
        return {"K_2pct": 0, "K_5pct": 0, "N_05": 0, "N_10": 0}
    e = e / e.sum()
    cum = np.cumsum(np.sort(e)[::-1])
    return {"K_2pct": int((e >= 0.02).sum()), "K_5pct": int((e >= 0.05).sum()),
            "N_05": int(np.searchsorted(cum, 0.95 - 1e-12) + 1), "N_10": int(np.searchsorted(cum, 0.90 - 1e-12) + 1)}


def efficiency(A, p, s, Gamma):
    return PG.programme_value(np.asarray(A, float), p, s) / Gamma


_VCACHE = {}


def eps_sets(p, s, Gamma):
    """{eps: (V (all dilates), K list)} for cached p; None otherwise."""
    key = (p, round(s, 9))
    if key in _VCACHE:
        return _VCACHE[key]
    if p not in PG.cached_primes():
        _VCACHE[key] = None
        return None
    reps, _, _ = PG.load_vertices(p)
    cost = PG.cost(reps, s)
    val = PG.coverage_min(reps, p) / cost ** s
    top = val.max()
    assert abs(top - Gamma) <= 1e-9 * Gamma, (p, s, top, Gamma)  # cache optimum == prediction
    out = {}
    for eps in EPS:
        R = reps[val >= (1 - eps) * Gamma]
        V = PG.expand_orbits(R, p)
        out[eps] = {"V": V, "K": (V > 0).sum(1), "n_orbits": int(len(R)), "n_vertices": int(len(V))}
    _VCACHE[key] = out
    return out


def eps_match(A, s, E):
    A = np.clip(np.asarray(A, float), 0, None)
    V = E["V"]
    cos = (V @ A) / (np.linalg.norm(V, axis=1) * np.linalg.norm(A) + 1e-300)
    w = MS.budget_shares(A, s)
    Wv = np.power(np.where(V > 0, V, 0.0), 1.0 / s); Wv = Wv / Wv.sum(1, keepdims=True)
    tv = 0.5 * np.abs(Wv - w[None, :]).sum(1)
    i = int(np.lexsort((tv, -cos))[0])  # registered rule: max cosine, ties -> smaller TV
    return {"cos_best": float(cos[i]), "tv_at_best_cos": float(tv[i]), "tv_min": float(tv.min()),
            "K_min": int(E["K"].min()), "K_max": int(E["K"].max()), "n_orbits": E["n_orbits"]}


def status_of(r):
    return r.get("csv_status") or r["status"]


def score_exp(exp):
    idx, P = alib.pred_index(exp)
    rows = []
    for r in alib.load(exp):
        st = status_of(r)
        if st not in ("ok", "not_converged") or "metrics" not in r:
            continue
        m = r["metrics"]["measurement"]
        if m.get("margin", -1) <= 0:
            continue
        c = r["config"]; nu = c["n"] + 1; p = c["p"]; s = round(nu / c["q"], 6)
        cell = idx.get((c["model"], nu, p, s))
        if cell is None:
            continue
        A = np.asarray(m["A"], float); G = cell["Gamma"]
        y0 = np.asarray(cell.get("y_star_canonical") or cell["y_star"], float)
        ec, ecp = energy_counts(A), energy_counts(y0)
        row = {"exp": exp, "run_id": r["run_id"], "status": st, "model": c["model"], "nu": nu, "p": p, "s": s, "q": c["q"],
               "seed": c["seed"], "K_star": cell["K_star"], "label": cell["label"],
               "runner_up_rel_gap": (cell.get("runner_up") or {}).get("rel_gap"),
               "eff": efficiency(A, p, s, G), "energy_clean": m["energy_frac"]["clean"],
               "thr_10": int(MS.counts(A, s)["thr_10"]), "thr_20": int(MS.counts(A, s)["thr_20"])}
        for k, v in ec.items():
            row[k] = v; row[k + "_pred_same_def"] = ecp[k]
        row["tv_to_ystar_registered"] = MT.score_amplitudes(A, nu / c["q"], np.asarray(cell.get("y_all") or [cell["y_star"]]))["share"]["tv"]
        E = eps_sets(p, nu / c["q"], G)
        for eps in EPS:
            if E is None:
                continue
            mm = eps_match(A, nu / c["q"], E[eps])
            for k, v in mm.items():
                row[f"eps{eps}_{k}"] = v
            row[f"eps{eps}_K2_in_range"] = bool(mm["K_min"] <= ec["K_2pct"] <= mm["K_max"])
        rows.append(row)
    return rows, P


def pooled(rows, key, pred="K_star", tol=1.0):
    meas = [r[key] for r in rows]; pr = [r[pred] for r in rows]
    out = MT.sweep_summary(meas, pr, tol)
    by = defaultdict(list)
    for r in rows:
        by[r["seed"]].append(abs(r[key] - r[pred]) <= tol)
    b = MT.bootstrap_over_seeds(list(by.values()), stat=np.mean, n_boot=2000)
    out["frac_within_tol_boot_ci_over_seeds"] = b["ci"]
    return out


def eff_summary(rows):
    e = np.array([r["eff"] for r in rows])
    by = defaultdict(list)
    for r in rows:
        by[r["seed"]].append(r["eff"])
    b = MT.bootstrap_over_seeds(list(by.values()), stat=np.median, n_boot=2000)
    return {"n": int(len(e)), "median": float(np.median(e)) if len(e) else None, "median_ci_over_seeds": b["ci"],
            "q10": float(np.percentile(e, 10)) if len(e) else None, "q90": float(np.percentile(e, 90)) if len(e) else None,
            "frac_ge_0.99": float(np.mean(e >= 0.99)) if len(e) else None, "frac_ge_0.95": float(np.mean(e >= 0.95)) if len(e) else None,
            "frac_ge_0.90": float(np.mean(e >= 0.90)) if len(e) else None}


def null_eff(exp, rng):
    """Efficiency of the two registered nulls: random allocation (support K*, Dirichlet shares; 1000 draws) and the
    step-0 random-init network (controls.json), and of the positive control (nu-block at y*)."""
    fn = os.path.join(ROOT, "experiments", exp, "analysis", "controls.json")
    C = json.load(open(fn)) if os.path.exists(fn) else None
    idx, _ = alib.pred_index(exp)
    out = {}
    for key, cell in idx.items():
        p, s, G, K = cell["p"], cell["s"], cell["Gamma"], cell["K_star"]
        h = (p - 1) // 2
        e = []
        for _ in range(1000):
            y = np.zeros(h); sup = rng.choice(h, K, replace=False); y[sup] = rng.dirichlet(np.ones(K)) ** s
            e.append(PG.programme_value(y, p, s) / G)
        rec = {"random_allocation_median": float(np.median(e)), "random_allocation_q95": float(np.percentile(e, 95))}
        E = eps_sets(p, s, G)
        if E is not None:  # the same eps-set share match applied to 50 random allocations (null for (iii))
            for eps in EPS:
                tvs, tvb = [], []
                for _ in range(50):
                    y = np.zeros(h); sup = rng.choice(h, K, replace=False); y[sup] = rng.dirichlet(np.ones(K)) ** s
                    mm = eps_match(y, s, E[eps]); tvs.append(mm["tv_min"]); tvb.append(mm["tv_at_best_cos"])
                rec[f"eps{eps}_random_allocation_median_tv_min"] = float(np.median(tvs))
                rec[f"eps{eps}_random_allocation_median_tv_at_best_cos"] = float(np.median(tvb))
        if C is not None:
            items = C if isinstance(C, list) else C.get("cells", C.get("controls", []))
            for it in (items if isinstance(items, list) else []):
                cc = it.get("cell", {})
                if (cc.get("model", "product"), cc.get("nu"), cc.get("p"), round(cc.get("s", -1), 6)) == key:
                    rec["random_init_net"] = PG.programme_value(np.asarray(it["null_init"]["measurement"]["A"]), p, s) / G
                    rec["positive_control"] = PG.programme_value(np.asarray(it["positive"]["measurement"]["A"]), p, s) / G
                    if E is not None:
                        for eps in EPS:
                            rec[f"eps{eps}_random_init_net_tv_min"] = eps_match(np.asarray(it["null_init"]["measurement"]["A"]), s, E[eps])["tv_min"]
        out["|".join(map(str, key))] = rec
    return out


def main():
    rng = np.random.default_rng(0)
    allsum = {"amendment": "A1 (post hoc)", "created": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "script": "experiments/common/amend_A1.py"}
    for exp in ("E1", "E6", "E7", "E8"):
        rows, P = score_exp(exp)
        alib.write_csv(rows, os.path.join(ROOT, "experiments", exp, "analysis", "A1_per_run.csv"))
        S = {"n_rows": len(rows)}
        for grp in ("ok", "not_converged"):
            R = [r for r in rows if r["status"] == grp]
            if not R:
                continue
            g = {"n": len(R), "efficiency": eff_summary(R)}
            for k in ("K_2pct", "K_5pct", "N_05", "N_10", "thr_10", "thr_20"):
                g[f"count_{k}_vs_Kstar"] = pooled(R, k)
            for k in ("K_2pct", "N_05"):
                g[f"count_{k}_vs_same_def_on_ystar"] = pooled(R, k, pred=k + "_pred_same_def")
            for eps in EPS:
                Rc = [r for r in R if f"eps{eps}_tv_min" in r]
                if Rc:
                    g[f"eps{eps}"] = {"n_cached": len(Rc),
                                      "frac_K2_in_eps_range": float(np.mean([r[f"eps{eps}_K2_in_range"] for r in Rc])),
                                      "median_tv_at_best_cos": float(np.median([r[f"eps{eps}_tv_at_best_cos"] for r in Rc])),
                                      "median_tv_min": float(np.median([r[f"eps{eps}_tv_min"] for r in Rc])),
                                      "median_cos_best": float(np.median([r[f"eps{eps}_cos_best"] for r in Rc])),
                                      "median_tv_to_ystar_registered_same_runs": float(np.median([r["tv_to_ystar_registered"] for r in Rc]))}
            # per cell
            cells = defaultdict(list)
            for r in R:
                cells[(r["model"], r["nu"], r["p"], r["s"])].append(r)
            g["cells"] = []
            for key in sorted(cells):
                C = cells[key]
                d = {"model": key[0], "nu": key[1], "p": key[2], "s": key[3], "n": len(C), "K_star": C[0]["K_star"],
                     "runner_up_rel_gap": C[0]["runner_up_rel_gap"], "label": C[0]["label"],
                     "median_K_2pct": float(np.median([r["K_2pct"] for r in C])), "median_N_05": float(np.median([r["N_05"] for r in C])),
                     "median_thr_10": float(np.median([r["thr_10"] for r in C])), "median_eff": float(np.median([r["eff"] for r in C])),
                     "min_eff": float(np.min([r["eff"] for r in C]))}
                if "eps0.01_K_min" in C[0]:
                    for eps in EPS:
                        d[f"eps{eps}_K_range"] = [C[0][f"eps{eps}_K_min"], C[0][f"eps{eps}_K_max"]]
                        d[f"eps{eps}_n_orbits"] = C[0][f"eps{eps}_n_orbits"]
                        d[f"eps{eps}_median_tv_min"] = float(np.median([r[f"eps{eps}_tv_min"] for r in C]))
                g["cells"].append(d)
            S[grp] = g
        S["nulls_and_positive"] = null_eff(exp, rng)
        json.dump(S, open(os.path.join(ROOT, "experiments", exp, "analysis", "A1_summary.json"), "w"), indent=1, default=float)
        allsum[exp] = {k: v for k, v in S.items() if k != "nulls_and_positive"}
        print(exp, len(rows), "rows", f"{time.time() - T0:.1f}s", flush=True)
    # E1b under A1: same (p, s), different nu
    rows, _ = score_exp("E1")
    ok = [r for r in rows if r["status"] == "ok"]
    by = defaultdict(list)
    for r in ok:
        by[(r["p"], r["s"])].append(r)
    e1b = []
    for (p, s), R in sorted(by.items()):
        nus = sorted({r["nu"] for r in R})
        if len(nus) < 2:
            continue
        med = {nu: (float(np.median([r["K_2pct"] for r in R if r["nu"] == nu])), float(np.median([r["eff"] for r in R if r["nu"] == nu]))) for nu in nus}
        e1b.append({"p": p, "s": s, "K_star": R[0]["K_star"], "by_nu_median_K2_eff": {str(k): v for k, v in med.items()},
                    "max_cross_nu_dK2": max(v[0] for v in med.values()) - min(v[0] for v in med.values())})
    # between-s spread at fixed (p, nu), for scale: |median K_2pct(s) - median K_2pct(s')| over all s != s'
    byc = defaultdict(list)
    for r in ok:
        byc[(r["p"], r["nu"], r["s"])].append(r["K_2pct"])
    med = {k: float(np.median(v)) for k, v in byc.items()}
    between = [abs(med[a] - med[b]) for a in med for b in med if a < b and a[:2] == b[:2]]
    within = []
    for (p, s), R in by.items():
        nus = sorted({r["nu"] for r in R})
        within += [abs(med[(p, a, s)] - med[(p, b, s)]) for i, a in enumerate(nus) for b in nus[i + 1:]]
    allsum["E1b"] = {"cells": e1b, "mean_abs_dK2_same_s_across_nu": float(np.mean(within)), "n_same_s_pairs": len(within),
                     "frac_same_s_pairs_within_1": float(np.mean(np.array(within) <= 1)),
                     "mean_abs_dK2_same_nu_across_s": float(np.mean(between)), "n_same_nu_pairs": len(between)}
    json.dump(allsum, open(os.path.join(HERE, "A1_summary.json"), "w"), indent=1, default=float)
    print("done", f"{time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
