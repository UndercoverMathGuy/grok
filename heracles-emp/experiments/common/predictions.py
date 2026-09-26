"""Assemble predictions.json for each experiment from core code (core.programme; pod-computed pred_*.json for p beyond
the vertex cache, which were produced by experiments/common/predict.py -> core.programme.solve_any on pod
grbou4dlnvjo6f). Light: cached-vertex lookups only. Usage: python experiments/common/predictions.py EXP"""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)
import numpy as np
from core import programme as P
from core import measure as MS

PRED = os.path.join(HERE, "pred")


def runner_up(p, s):
    """Programme value of the best vertex CLASS that is not optimal (relative gap), cached p only."""
    reps, sizes, meta = P.load_vertices(p)
    c = P.cost(reps, s)
    cmin = c.min()
    rest = c[c > cmin * (1 + 1e-9)]
    if len(rest) == 0:
        return None
    c2 = rest.min()
    i2 = int(np.flatnonzero(c == c2)[0])
    return {"gamma_runner_up": float(c2 ** (-s)), "rel_gap": float(1 - (c2 ** (-s)) / (cmin ** (-s))),
            "K_runner_up": int((reps[i2] > 0).sum())}


def pred(p, s):
    fn = os.path.join(PRED, f"pred_p{p}_s{s:g}.json")
    r = json.load(open(fn))
    out = {"p": p, "s": s, "Gamma": r["gamma"], "K_star": r["K"], "K_all": r["K_all"], "label": r["label"],
           "y_star_canonical": r["y"], "y_all": r["y_all"], "tight_offsets_canonical": r["tight_offsets"][0] if r["tight_offsets"] else None,
           "source": os.path.relpath(fn, ROOT)}
    if p in P.cached_primes():
        out["runner_up"] = runner_up(p, s)
    elif "best_found" in r:
        d = r["best_found"]["distinct"]
        if len(d) > 1:
            out["runner_up"] = {"gamma_runner_up": d[1][0], "rel_gap": 1 - d[1][0] / d[0][0], "K_runner_up": d[1][1],
                                "note": "second-best local minimum found by best_found (not certified)"}
        out["kcap"] = r.get("kcap")
    return out


def ceil(p, nu, q, G):
    Bp = 2.0 * (2.0 * p * nu) ** (-nu / 2.0)
    return {"B_p": Bp, "universal_product": Bp * G, "universal_power": math.factorial(nu - 1) * Bp * G,
            "class_optimum_IF_product": nu ** (-(nu - q) / q) * Bp * G}


def e1():
    sys.path.insert(0, HERE)
    import build
    cells = []
    for nu in (3, 4, 5):
        for p in build.E1_P[nu]:
            for s in build.E1_S[nu]:
                q = nu / s
                pr = pred(p, s)
                pr.update({"nu": nu, "q": q, "ceilings": ceil(p, nu, q, pr["Gamma"])})
                cells.append(pr)
    return {"experiment": "E1", "cells": cells}


def e3():
    cells = []
    for model, nu, ps in [("product", 3, [23, 59, 113]), ("product", 4, [23, 43, 59]), ("product", 5, [11, 13, 17]),
                          ("power", 3, [23, 59, 113])]:
        for p in ps:
            h = (p - 1) // 2
            G = p / (p - 1)  # cor:dense, Gamma_1(p) = p/(p-1), unique maximiser y_j = 2/(p-1)
            c = {"model": model, "nu": nu, "p": p, "q": float(nu), "s": 1.0, "Gamma": G, "K_star": h,
                 "y_star": [2.0 / (p - 1)] * h, "label": "closed form (cor:dense)", "ceilings": ceil(p, nu, float(nu), G)}
            if p in P.cached_primes():
                sol = P.solve(p, 1.0)
                c["crosscheck_exact_cache"] = {"gamma": sol.gamma, "rel_diff": abs(sol.gamma - G) / G,
                                               "K_all": sorted(set(sol.kstar_all.tolist()))}
            cells.append(c)
    return {"experiment": "E3", "cells": cells,
            "note": "power nu=3 (n=2, q=3): ceiling 2 B_p p/(p-1) attained (Morwani construction). Product q=nu: nu-block attains B_p Gamma_1."}


def e7(p=37, nu=4):
    sw = json.load(open(os.path.join(PRED, f"switch_p{p}.json")))
    qg = json.load(open(os.path.join(ROOT, "experiments", "E7", "q_grid.json")))
    cells = []
    for s in qg["s"]:
        q = nu / s
        sol = P.solve(p, s)
        ru = runner_up(p, s)
        cells.append({"p": p, "nu": nu, "q": q, "s": s, "Gamma": sol.gamma, "K_star": sol.kstar,
                      "K_all": sorted(set(sol.kstar_all.tolist())), "y_star_canonical": sol.y.tolist(),
                      "y_all": sol.y_all.tolist(), "runner_up": ru, "label": "exact (vertex cache)",
                      "ceilings": ceil(p, nu, q, sol.gamma)})
    return {"experiment": "E7", "switch_points": sw["switches"], "plateaus": sw["plateaus"], "cells": cells,
            "source_switch": f"experiments/common/pred/switch_p{p}.json"}


def e6():
    pr = pred(23, 1.5)
    h = 11
    return {"experiment": "E6", "cell": {**pr, "nu": 3, "q": 2.0, "ceilings": ceil(23, 3, 2.0, pr["Gamma"])},
            "equivariance": "relabelled init a -> u a gives A'_j = A_{fold(u j)} exactly in exact arithmetic (GD, full batch)",
            "marginal_prediction": "P(j active) = E[K]/h for every j (h = 11), under the dilation-invariant N(0,1) init law"}


def e8():
    sys.path.insert(0, HERE)
    cells = []
    for p in sorted(set([11, 13, 17, 19, 29, 37, 47, 71] + [23, 31, 43, 59, 83, 113])):
        for s in (1.25, 1.5):
            pr = pred(p, s)
            pr.update({"nu": 3, "q": 3 / s, "K_over_log_p": pr["K_star"] / math.log(p), "ceilings": ceil(p, 3, 3 / s, pr["Gamma"])})
            cells.append(pr)
    return {"experiment": "E8", "cells": cells}


if __name__ == "__main__":
    exp = sys.argv[1]
    f = {"E1": e1, "E3": e3, "E7": e7, "E6": e6, "E8": e8}[exp]
    out = f()
    out["generated_by"] = "experiments/common/predictions.py (core.programme sha256 " + P.SELF_SHA256 + ")"
    os.makedirs(os.path.join(ROOT, "experiments", exp), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "experiments", exp, "predictions.json"), "w"), indent=1)
    print(exp, "cells", len(out.get("cells", [out.get("cell")])))
