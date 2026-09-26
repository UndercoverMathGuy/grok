"""AUDIT: independent brute-force solve of the share programme at small p (own code), vs predictions.json and
core.programme (imported only to compare Gamma / y* / vertex data, as allowed).
Gamma_s(p) = max { min_{d=1..h} sum_j y_j (1 - cos(2 pi j d / p)) : y >= 0, sum y_j^{1/s} <= 1 }.
With v >= 0, Av >= 1: C_s = min sum v^{1/s}, Gamma = C^{-s}. The cost is concave for s >= 1, so the minimum is at a
vertex of the polyhedron; enumerate every basis (h tight constraints out of the 2h: A_d v = 1 or v_j = 0)."""
import itertools, json, os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def vertices(p):
    h = (p - 1) // 2
    d = np.arange(1, h + 1); j = np.arange(1, h + 1)
    A = 1 - np.cos(2 * np.pi * np.outer(d, j) / p)
    rows = np.vstack([A, np.eye(h)]); rhs = np.concatenate([np.ones(h), np.zeros(h)])
    V = []
    for S in itertools.combinations(range(2 * h), h):
        M = rows[list(S)]
        if abs(np.linalg.det(M)) < 1e-10: continue
        v = np.linalg.solve(M, rhs[list(S)])
        if (v >= -1e-10).all() and (A @ v >= 1 - 1e-9).all():
            V.append(np.clip(v, 0, None))
    V = np.unique(np.round(np.array(V), 10), axis=0)
    return A, V

def solve(p, s, A, V):
    cost = (V ** (1 / s)).sum(1)
    i = np.argsort(cost)
    C = cost[i[0]]; G = C ** (-s)
    opt = V[np.abs(cost - C) <= 1e-9 * C]
    ystar = opt / C ** s
    second = cost[cost > C * (1 + 1e-9)].min() if (cost > C * (1 + 1e-9)).any() else None
    return G, ystar, (None if second is None else 1 - (C / second) ** s)

if __name__ == "__main__":
    t0 = time.time()
    primes = [int(x) for x in sys.argv[1:]] or [5, 7, 11, 13]
    preds = {}
    for e in ("E1", "E7", "E8"):
        P = json.load(open(os.path.join(ROOT, "experiments", e, "predictions.json")))
        for c in (P.get("cells") or [P["cell"]]):
            preds[(c["p"], round(c["s"], 6))] = c
    sys.path.insert(0, ROOT)
    from core import programme as PG
    out = []
    for p in primes:
        A, V = vertices(p)
        for s in (1.0, 1.25, 1.5, 2.0, 2.5):
            G, Y, gap = solve(p, s, A, V)
            K = sorted(set((Y > 1e-9).sum(1).tolist()))
            rec = {"p": p, "s": s, "Gamma_brute": G, "K_brute": K, "n_opt_vertices": len(Y), "runner_up_gap_brute": gap}
            c = preds.get((p, s))
            if c:
                rec["Gamma_pred"] = c["Gamma"]; rec["K_pred"] = c["K_star"]
                Yp = np.array(c.get("y_all") or [c["y_star"]])
                rec["y_match"] = bool(min(np.abs(Y[:, None, :] - Yp[None]).max(-1).min(0).max(), 1e9) < 1e-7) if Yp.shape[1] == Y.shape[1] else None
                rec["gap_pred"] = (c.get("runner_up") or {}).get("rel_gap")
            try:
                rec["Gamma_core_value_at_brute_ystar"] = float(PG.programme_value(Y[0], p, s))
            except Exception as ex:
                rec["core_err"] = repr(ex)
            if s == 1.0: rec["closed_form_p_over_pm1"] = p / (p - 1)
            out.append(rec); print(rec)
    json.dump(out, open(os.path.join(ROOT, "audit", f"brute_programme_{'_'.join(map(str, primes))}.json"), "w"), indent=1)
    print(f"{time.time()-t0:.1f}s")
