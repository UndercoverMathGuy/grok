"""Amendment A3 (POST HOC, not preregistered): best-known programme optimum at cells where an independent, earlier
search found a HIGHER value than the preregistered best-found prediction. Gamma_s(p) is a maximum, so the higher value
is the better estimate; nothing about trained networks is used.

Source of the alternative allocations: the author's earlier support search (heracles/FINAL_RUNS/LADDER/json/cells.json,
field rows[K*].free: support S and budget shares b). They are copied below verbatim and RE-EVALUATED here with
heracles-emp's own core.programme.programme_value, so the value used is computed by this directory's code.

Affected cells (the only ones where both searches exist and they differ):
  p = 113, s = 1.5 (nu = 3, b = 2): prediction K* = 6, Gamma = 0.269864 -> best known K* = 7, Gamma = 0.275936
  p = 59,  s = 2   (nu = 4, b = 2): prediction K* = 4, Gamma = 0.146762 -> best known K* = 4, Gamma = 0.152806
At p = 59 and 71, s = 1.5 the two searches agree to 1e-6.
Writes experiments/common/A3_bestknown.json and prints the paper numbers that change.
Usage: uv run python experiments/common/amend_A3_bestknown.py   (seconds)
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np
from core import programme as PG

ALT = {  # (p, s): support (frequencies 1..h) and budget shares b_j = y_j^(1/s), copied from LADDER cells.json
    (113, 1.5): {"S": [1, 6, 16, 31, 35, 44, 55],
                 "b": [0.1487478674182879, 0.17193007187207457, 0.13086638919473642, 0.13263210500181966,
                       0.13525683333114755, 0.109739929496645, 0.17082680368528896], "ladder_val": 0.2759357645841436},
    (59, 2.0): {"S": [1, 7, 20, 24],
                "b": [0.29206580140489297, 0.23587049737164006, 0.20464637837013966, 0.26741732285332737],
                "ladder_val": 0.1528062257159834},
}


def main():
    out = {"amendment": "A3 (post hoc)", "script": "experiments/common/amend_A3_bestknown.py", "cells": {}}
    for (p, s), a in ALT.items():
        h = (p - 1) // 2
        y = np.zeros(h)
        y[np.array(a["S"]) - 1] = np.array(a["b"]) ** s
        val = PG.programme_value(y, p, s)
        pred = json.load(open(os.path.join(HERE, "pred", f"pred_p{p}_s{s:g}.json")))
        assert val > pred["gamma"], (p, s, val, pred["gamma"])
        assert abs(val - a["ladder_val"]) < 1e-9, (val, a["ladder_val"])
        out["cells"][f"{p}|{s}"] = {"p": p, "s": s, "Gamma_best_known": float(val), "K_best_known": len(a["S"]),
                                    "support": a["S"], "Gamma_prediction": pred["gamma"], "K_prediction": pred["K"],
                                    "rel_gain": float(val / pred["gamma"] - 1)}
        print(f"p={p} s={s}: re-evaluated {val:.6f} (LADDER {a['ladder_val']:.6f}) vs prediction {pred['gamma']:.6f} K {pred['K']} -> K {len(a['S'])}")
    # ---- paper numbers under the best-known optimum (converged runs; thr_20 unless noted)
    import csv
    import amend_A2_shape as A2
    ov = {(c["p"], c["s"]): c for c in out["cells"].values()}
    rows = A2.read("E1") + A2.read("E8") + A2.read("E7")
    ok = [r for r in rows if r["status"] == "ok"]
    for r in ok:
        c = ov.get((r["p"], r["s"]))
        if c and not (r["p"] == 113 and r["nu"] != 3) and not (r["p"] == 59 and r["s"] == 2.0 and r["nu"] != 4):
            r["K_star"] = c["K_best_known"]
    e1 = [r for r in ok if r["exp"] == "E1"]
    k = np.array([r["K_star"] for r in e1]); m = np.array([r["thr_20"] for r in e1])
    within = float(np.mean(np.abs(m - k) <= 1))
    best_const = max(float(np.mean(np.abs(m - c) <= 1)) for c in range(0, 20))
    out["pooled_E1_thr20_within1"] = within; out["pooled_E1_best_constant_within1"] = best_const
    # efficiency: eff = value / Gamma, so rescale by Gamma_prediction / Gamma_best_known in the two cells
    A1 = [r for r in csv.DictReader(open(os.path.join(ROOT, "experiments", "E1", "analysis", "A1_per_run.csv"))) if r["status"] == "ok"]
    eff = []
    for r in A1:
        c = ov.get((int(r["p"]), round(float(r["s"]), 6)))
        f = c["Gamma_prediction"] / c["Gamma_best_known"] if c else 1.0
        eff.append(float(r["eff"]) * f)
    eff = np.array(eff)
    out["E1_eff_median"] = float(np.median(eff)); out["E1_eff_frac_ge_0.95"] = float(np.mean(eff >= 0.95))
    N = json.load(open(os.path.join(ROOT, "experiments", "E1", "analysis", "A1_summary.json")))["nulls_and_positive"]
    ra, ri = [], []
    for key, v in N.items():
        _, nu, p_, s_ = key.split("|"); c = ov.get((int(p_), round(float(s_), 6)))
        f = c["Gamma_prediction"] / c["Gamma_best_known"] if c else 1.0
        ra.append(v["random_allocation_median"] * f)
        if v.get("random_init_net") is not None:
            ri.append(v["random_init_net"] * f)
    out["E1_null_random_allocation_median"] = float(np.median(ra)); out["E1_null_random_init_median"] = float(np.median(ri))
    # direction test T1 (E1 + E8) under every count definition
    rng = np.random.default_rng(20260926)
    e18 = [r for r in ok if r["exp"] in ("E1", "E8")]
    out["T1_E1E8"] = {}
    for col in A2.COUNTS:
        t, _ = A2.direction(e18, ["nu", "p"], "s", col, rng)
        out["T1_E1E8"][col] = t
    t = out["T1_E1E8"]
    print(f"pooled thr_20 within 1: {within:.3f} (best constant {best_const:.3f})")
    print(f"E1 eff median {out['E1_eff_median']:.4f}, frac>=0.95 {out['E1_eff_frac_ge_0.95']:.3f}, nulls {out['E1_null_random_allocation_median']:.3f} / {out['E1_null_random_init_median']:.3f}")
    print("T1 thr_20:", {k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in t["thr_20"].items()})
    print("T1 across definitions: right", min(v["right"] for v in t.values()), "-", max(v["right"] for v in t.values()),
          "slope", round(min(v["slope_through_origin"] for v in t.values()), 2), "-", round(max(v["slope_through_origin"] for v in t.values()), 2),
          "p max", max(v["p_one_sided"] for v in t.values()), "null", round(min(v["null_mean"] for v in t.values()), 2), "-", round(max(v["null_mean"] for v in t.values()), 2))
    # ---- same s, different degree: signed comparison (post hoc; requested by the author 2026-09-26)
    import itertools
    from math import comb
    cells = {}
    for r in ok:
        cells.setdefault((r["p"], r["s"], r["nu"]), []).append(r)
    med = {k2: (float(np.median([x["thr_20"] for x in v])), v[0]["K_star"]) for k2, v in cells.items()}
    diffs = []
    for (p_, s_) in sorted({(k2[0], k2[1]) for k2 in med}):
        nus = sorted(k2[2] for k2 in med if k2[0] == p_ and k2[1] == s_)
        for lo, hi in itertools.combinations(nus, 2):
            diffs.append(med[(p_, s_, hi)][0] - med[(p_, s_, lo)][0])
    d = np.array(diffs); nz = int((d != 0).sum()); k_ = int((d < 0).sum())
    sign_p = 2 * sum(comb(nz, i) for i in range(max(k_, nz - k_), nz + 1)) / 2 ** nz
    by_nu = {}
    for nu in (3, 4, 5):
        e = [med[k2][0] - med[k2][1] for k2 in med if k2[2] == nu and k2[1] in (1.25, 1.5) and k2[0] <= 43]
        by_nu[str(nu)] = {"mean_count_minus_Kstar": float(np.mean(e)), "n_cells": len(e)}
    eff_nu = {}
    for nu in ("3", "4", "5"):
        ee = [float(r["eff"]) * ((ov[(int(r["p"]), round(float(r["s"]), 6))]["Gamma_prediction"] / ov[(int(r["p"]), round(float(r["s"]), 6))]["Gamma_best_known"]) if (int(r["p"]), round(float(r["s"]), 6)) in ov else 1.0)
              for r in A1 if r["nu"] == nu]
        eff_nu[nu] = float(np.median(ee))
    out["same_s_signed"] = {"n_pairs": len(d), "within_1": int((np.abs(d) <= 1).sum()), "higher_nu_fewer": k_,
                            "equal": int((d == 0).sum()), "higher_nu_more": int((d > 0).sum()), "mean_diff_hi_minus_lo": float(d.mean()),
                            "sign_test_two_sided_p": float(sign_p), "count_minus_Kstar_by_nu_s_le_1.5_p_le_43": by_nu,
                            "E1_eff_median_by_nu": eff_nu}
    print("same s:", out["same_s_signed"])
    # direction test including non-converged runs (selection check requested after the training-code audit, M1)
    allt = [r for r in rows if r["status"] in ("ok", "not_converged") and r["exp"] in ("E1", "E8")]
    for r in allt:
        c = ov.get((r["p"], r["s"]))
        if c:
            r["K_star"] = c["K_best_known"]
    t_all, _ = A2.direction(allt, ["nu", "p"], "s", "thr_20", np.random.default_rng(20260926))
    out["T1_all_trained_thr20"] = t_all
    print("T1 all trained thr_20:", {k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in t_all.items()})
    json.dump(out, open(os.path.join(HERE, "A3_bestknown.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
