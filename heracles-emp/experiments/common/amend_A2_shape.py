"""Amendment A2 (POST HOC, not preregistered): shape tests of the count along the sweeps, requested by the author on
2026-09-26 after seeing A1 and the audit (the pooled "count = K*" comparison is weak against a constant baseline).

A constant or any predictor that ignores s scores zero on a direction test, so these tests ask whether the measured
count moves the way K* moves when one knob is changed and the others are held fixed:
  T1  q sweep at fixed (nu, p): for every pair of cells in the same (nu, p) whose K* differ, does the median count
      move in the same direction as K*? (E1 + E8 pooled; E7 fine grid separately.)
  T2  nu ladder at fixed q = 2 (plain weight decay), fixed p: same direction test across nu (E1, E8 and the E7 q = 2 cell).
  T3  same s across nu (the "s alone decides" claim): mean |difference of median counts| between cells with the same
      (p, s) and different nu (K* identical by construction), against the same quantity between cells with the same
      (p, nu) and different s.
  T4  clean energy fraction against s (descriptive): Spearman within nu, converged runs, plus E3 at s = 1.
Null for T1/T2: the measured counts are shuffled across the cells of each group at the seed level (B = 1000), keeping
K* fixed; reported as the mean and 95th percentile of the null agreement fraction and a one-sided p-value.
Every count definition is scored (NORMS E16): the 9 registered ones (from the preregistered per_run.csv) and the 4
post-hoc ones of A1. Converged runs (status ok) only; not-converged runs
are scored separately under "not_converged".
Reads experiments/{E1,E7,E8}/analysis/A1_per_run.csv and experiments/E3/analysis/per_run.csv (no run directories).
Writes experiments/common/A2_summary.json and experiments/common/A2_pairs.csv.
Usage: uv run python experiments/common/amend_A2_shape.py   (seconds)
"""
import csv, itertools, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
REGISTERED = ["thr_20", "thr_10", "thr_5", "thr_1", "pr_amp", "pr_budget", "mass_90", "mass_95", "mass_99"]  # core.measure.COUNT_KEYS
POSTHOC = ["K_2pct", "K_5pct", "N_05", "N_10"]  # Amendment A1
COUNTS = REGISTERED + POSTHOC
B = 1000


def read(exp):
    rows = list(csv.DictReader(open(os.path.join(ROOT, "experiments", exp, "analysis", "A1_per_run.csv"))))
    reg = {r["run_id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "experiments", exp, "analysis", "per_run.csv")))}
    for r in rows:  # registered counts from the preregistered per-run table (same run ids)
        for k in REGISTERED:
            v = reg[r["run_id"]][f"cnt_{k}"]
            assert k not in r or float(r[k]) == float(v), (r["run_id"], k)
            r[k] = v
    out = []
    for r in rows:
        d = {"exp": exp, "status": r["status"], "nu": int(r["nu"]), "p": int(r["p"]), "s": round(float(r["s"]), 6),
             "q": round(float(r["q"]), 6), "seed": r["seed"], "K_star": int(r["K_star"]), "energy_clean": float(r["energy_clean"])}
        for k in COUNTS:
            d[k] = float(r[k])
        out.append(d)
    return out


def cells(rows, group, vary, col):
    """{group key: {vary value: (K*, median count, n)}}"""
    G = {}
    for r in rows:
        G.setdefault(tuple(r[g] for g in group), {}).setdefault(r[vary], []).append(r)
    out = {}
    for gk, by in G.items():
        out[gk] = {v: (rs[0]["K_star"], float(np.median([x[col] for x in rs])), len(rs)) for v, rs in by.items()}
    return out


def direction(rows, group, vary, col, rng=None):
    C = cells(rows, group, vary, col)
    pairs = []
    for gk, by in C.items():
        for a, b in itertools.combinations(sorted(by), 2):
            dk, dm = by[b][0] - by[a][0], by[b][1] - by[a][1]
            if dk != 0:
                pairs.append({"group": gk, "a": a, "b": b, "dK_star": dk, "dmeas": dm, "n_a": by[a][2], "n_b": by[b][2]})
    n = len(pairs)
    right = sum(np.sign(x["dK_star"]) == np.sign(x["dmeas"]) for x in pairs)
    tie = sum(x["dmeas"] == 0 for x in pairs)
    dk = np.array([x["dK_star"] for x in pairs], float); dm = np.array([x["dmeas"] for x in pairs], float)
    slope0 = float(dk @ dm / (dk @ dk)) if n else None  # least squares through the origin (differences)
    res = {"n_pairs": n, "right": int(right), "tie": int(tie), "wrong": int(n - right - tie),
           "frac_right": right / n if n else None, "slope_through_origin": slope0}
    if rng is not None and n:
        # vectorised: per group, cells as index sets into the group's count vector
        gidx = {}
        for i, r in enumerate(rows):
            gidx.setdefault(tuple(r[g] for g in group), []).append(i)
        vals = np.array([r[col] for r in rows], float)
        spec = []  # (group indices, [(cell mask a, cell mask b, sign dK*)])
        for gk, ii in gidx.items():
            ii = np.array(ii); vs = np.array([rows[i][vary] for i in ii]); ks = {v: rows[ii[vs == v][0]]["K_star"] for v in set(vs)}
            prs = [(vs == a, vs == b, np.sign(ks[b] - ks[a])) for a, b in itertools.combinations(sorted(ks), 2) if ks[b] != ks[a]]
            spec.append((ii, prs))
        null = np.empty(B)
        for k in range(B):
            hit = 0
            for ii, prs in spec:
                v = rng.permutation(vals[ii])
                for ma, mb, sg in prs:
                    hit += np.sign(np.median(v[mb]) - np.median(v[ma])) == sg
            null[k] = hit / n
        res.update({"null_mean": float(null.mean()), "null_q95": float(np.quantile(null, 0.95)),
                    "p_one_sided": float((1 + (null >= res["frac_right"]).sum()) / (B + 1))})
    return res, pairs


def mean_abs_diff(rows, group, vary, col):
    C = cells(rows, group, vary, col)
    d = [abs(by[b][1] - by[a][1]) for by in C.values() for a, b in itertools.combinations(sorted(by), 2)]
    dk = [abs(by[b][0] - by[a][0]) for by in C.values() for a, b in itertools.combinations(sorted(by), 2)]
    return {"n_pairs": len(d), "mean_abs_dmeas": float(np.mean(d)) if d else None,
            "mean_abs_dK_star": float(np.mean(dk)) if dk else None, "frac_within_1": float(np.mean(np.array(d) <= 1)) if d else None}


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    rng = np.random.default_rng(20260926)
    allrows = read("E1") + read("E8") + read("E7")
    out = {"amendment": "A2 (post hoc)", "script": "experiments/common/amend_A2_shape.py", "B_null": B}
    pair_rows = []
    for status in ("ok", "not_converged"):
        R = [r for r in allrows if r["status"] == status]
        e18 = [r for r in R if r["exp"] in ("E1", "E8")]; e7 = [r for r in R if r["exp"] == "E7"]
        S = {}
        for col in COUNTS:
            g = rng if status == "ok" else None  # shuffle null for converged runs only (laptop time limit)
            t1, p1 = direction(e18, ["nu", "p"], "s", col, g)
            t1e7, p7 = direction(e7, ["nu", "p"], "s", col, g)
            q2 = [r for r in R if abs(r["q"] - 2.0) < 1e-6]  # E1 + E8 + the E7 q = 2 cell
            t2, p2 = direction(q2, ["p"], "nu", col, g)
            t3 = {"same_s_across_nu": mean_abs_diff(e18, ["p", "s"], "nu", col),
                  "same_nu_across_s": mean_abs_diff(e18, ["p", "nu"], "s", col)}
            S[col] = {"T1_qsweep_E1E8": t1, "T1_qsweep_E7": t1e7, "T2_nu_ladder_q2": t2, "T3_same_s": t3}
            if status == "ok":
                for tag, P in (("T1_E1E8", p1), ("T1_E7", p7), ("T2_q2", p2)):
                    for x in P:
                        pair_rows.append({"count": col, "test": tag, "group": "|".join(map(str, x["group"])), "a": x["a"], "b": x["b"],
                                          "dK_star": x["dK_star"], "dmeas": x["dmeas"], "n_a": x["n_a"], "n_b": x["n_b"]})
        # T4
        T4 = {}
        for nu in sorted({r["nu"] for r in R}):
            rs = [r for r in R if r["nu"] == nu and r["exp"] in ("E1", "E8", "E7")]
            if len({r["s"] for r in rs}) > 1:
                T4[f"nu{nu}"] = {"n": len(rs), "spearman_clean_vs_s": spearman([r["s"] for r in rs], [r["energy_clean"] for r in rs]),
                                 "median_clean_by_s": {str(s): float(np.median([r["energy_clean"] for r in rs if r["s"] == s])) for s in sorted({r["s"] for r in rs})}}
        S["T4_clean_vs_s"] = T4
        out[status] = S
    e3 = [r for r in csv.DictReader(open(os.path.join(ROOT, "experiments", "E3", "analysis", "per_run.csv"))) if r["status"] == "ok"]
    out["ok"]["T4_clean_vs_s"]["E3_q_eq_nu"] = {"n": len(e3), "median_clean": float(np.median([float(r["energy_clean"]) for r in e3])),
                                               "min_clean": float(min(float(r["energy_clean"]) for r in e3))}
    json.dump(out, open(os.path.join(HERE, "A2_summary.json"), "w"), indent=1)
    with open(os.path.join(HERE, "A2_pairs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys())); w.writeheader(); w.writerows(pair_rows)
    for col in COUNTS:
        o = out["ok"][col]
        print(f"{col:7s} T1 E1+E8 {o['T1_qsweep_E1E8']['right']}/{o['T1_qsweep_E1E8']['n_pairs']} (tie {o['T1_qsweep_E1E8']['tie']}) "
              f"null {o['T1_qsweep_E1E8']['null_mean']:.2f}/{o['T1_qsweep_E1E8']['null_q95']:.2f} p={o['T1_qsweep_E1E8']['p_one_sided']:.4f} slope {o['T1_qsweep_E1E8']['slope_through_origin']:.2f} | "
              f"E7 {o['T1_qsweep_E7']['right']}/{o['T1_qsweep_E7']['n_pairs']} null {o['T1_qsweep_E7']['null_mean']:.2f}/{o['T1_qsweep_E7']['null_q95']:.2f} p={o['T1_qsweep_E7']['p_one_sided']:.4f} slope {o['T1_qsweep_E7']['slope_through_origin']:.2f} | "
              f"T2 {o['T2_nu_ladder_q2']['right']}/{o['T2_nu_ladder_q2']['n_pairs']} (tie {o['T2_nu_ladder_q2']['tie']}) p={o['T2_nu_ladder_q2']['p_one_sided']:.3f} | "
              f"T3 same-s {o['T3_same_s']['same_s_across_nu']['mean_abs_dmeas']:.2f} (n {o['T3_same_s']['same_s_across_nu']['n_pairs']}) vs across-s {o['T3_same_s']['same_nu_across_s']['mean_abs_dmeas']:.2f} (|dK*| {o['T3_same_s']['same_nu_across_s']['mean_abs_dK_star']:.2f})")
    print(json.dumps(out["ok"]["T4_clean_vs_s"], indent=0)[:900])


if __name__ == "__main__":
    main()
