"""AUDIT check 3: E3, E4, E5, E6, E7, E8 headline numbers + RESULTS.md cross-experiment numbers (own code)."""
import itertools, json, math, os, sys
from collections import defaultdict
import numpy as np, torch
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_audit import *
out = {}
def r3(x): return round(float(x), 3)

# ---------------- E3
R = runs("E3"); idx, _ = preds("E3")
out["E3_status"] = dict(__import__("collections").Counter(r["status"] for r in R))
cells = defaultdict(list)
for r in R:
    if r["status"] not in ("ok", "not_converged") or r["m"] is None or r["m"]["margin"] <= 0: continue
    c = r["c"]; p = c["p"]; nu = c["n"] + 1; A = np.asarray(r["m"]["A"]); h = len(A); w = A / A.sum()
    ceil = Bp(p, nu) * p / (p - 1) * (math.factorial(c["n"]) if c["model"] == "power" else 1)
    x = {"thr10": thr(A, .1), "mm": A.min() / A.max(), "cv": A.std() / A.mean(), "tvu": 0.5 * np.abs(w - 1 / h).sum(), "g": r["m"]["gamma"] / ceil, "h": h}
    x["match"] = x["thr10"] == h and x["mm"] >= .5 and x["tvu"] <= .05
    cells[(c["model"], nu, p, r["status"])].append(x)
E3 = []
for k in sorted(cells):
    g = cells[k]; med = lambda z: float(np.median([x[z] for x in g]))
    E3.append({"cell": k, "n": len(g), "thr10": med("thr10"), "h": g[0]["h"], "minmax": r3(med("mm")), "cv": r3(med("cv")), "tvu": round(med("tvu"), 4),
               "g_ceil": r3(med("g")), "frac_match": r3(np.mean([x["match"] for x in g])),
               "supported": med("thr10") == g[0]["h"] and med("mm") >= .5 and med("tvu") <= .05})
out["E3"] = E3

# ---------------- E4 (uses E1 runs)
rng = np.random.default_rng(0)
recs = []
for r in runs("E1"):
    m = r["m"]
    if m is None or r["status"] not in ("ok", "not_converged") or m["margin"] <= 0: continue
    s = m["s"]; w = shares(m["A"], s)
    rl = 1 - np.asarray(m["ablation"]["h_ablated"]) / m["ablation"]["h_base"]
    rl_check = float(np.abs(rl - np.asarray(m["ablation"]["rel_loss"])).max())
    b = 1 - (1 - w) ** s
    perm = []
    for _ in range(100):
        wp = rng.permutation(w); sel = wp >= .01; perm.append(np.mean((rl >= 1 - (1 - wp) ** s - 1e-9)[sel]))
    recs.append({"id": r["run_id"], "st": r["status"], "nu": m["nu"], "w": w, "rl": rl, "b": b, "perm": np.mean(perm), "rlchk": rl_check})
E4 = {"max_relloss_recompute_diff": max(x["rlchk"] for x in recs)}
for g in ("ok", "not_converged"):
    G = [x for x in recs if x["st"] == g]
    sat = sum(int((x["rl"] >= x["b"] - 1e-9)[x["w"] >= .01].sum()) for x in G); tot = sum(int((x["w"] >= .01).sum()) for x in G)
    satall = sum(int((x["rl"] >= x["b"] - 1e-9).sum()) for x in G); totall = sum(len(x["w"]) for x in G)
    exc = np.concatenate([(x["rl"] - x["b"])[x["w"] >= .01] for x in G])
    viol = [(x["id"], int(j + 1), r3(x["w"][j]), r3(x["rl"][j]), r3(x["b"][j])) for x in G for j in np.flatnonzero((x["w"] >= .01) & (x["rl"] < x["b"] - 1e-9))]
    bynu = {nu: r3(sum(int((x["rl"] >= x["b"] - 1e-9)[x["w"] >= .01].sum()) for x in G if x["nu"] == nu) / max(1, sum(int((x["w"] >= .01).sum()) for x in G if x["nu"] == nu))) for nu in (3, 4, 5)}
    E4[g] = {"runs": len(G), "sat": sat, "tot": tot, "frac": r3(sat / tot), "all_j": r3(satall / totall), "perm_null": r3(np.mean([x["perm"] for x in G])),
             "excess_q": {q: r3(np.percentile(exc, q)) for q in (1, 5, 50)}, "by_nu": bynu, "violations": viol}
out["E4"] = E4

# ---------------- E5 (uses E1 runs)
idx1, _ = preds("E1")
def near(hf, eps):
    hf = np.asarray(hf); mn = hf.min(); return set((np.flatnonzero(hf <= mn + eps * abs(mn)) + 1).tolist())
rows = []
for r in runs("E1"):
    m = r["m"]
    if m is None or r["status"] not in ("ok", "not_converged") or m["margin"] <= 0: continue
    cell = idx1[key_of(r)]; p = m["p"]; s = m["s"]; A = np.asarray(m["A"])
    Y = np.asarray(cell["y_all"]); y = Y[align(A, Y, s)]
    covy = cov_matrix(p) @ y; tight = set((np.flatnonzero(covy <= covy.min() * (1 + 1e-9)) + 1).tolist())
    hf = m["per_offset_margin_folded"]; K = thr(A, .1)
    S5, S1 = near(hf, .05), near(hf, .01)
    j5 = len(S5 & tight) / len(S5 | tight); j1 = len(S1 & tight) / len(S1 | tight)
    rows.append({"st": r["status"], "match": len(S5) >= K and j5 >= .5, "ge": len(S5) >= K, "j5": j5, "j1": j1, "n1_eq": len(S1) == len(tight),
                 "d1": len(S1) - len(tight), "d5": len(S5) - K})
E5 = {}
for g in ("ok", "not_converged"):
    G = [x for x in rows if x["st"] == g]
    E5[g] = {"n": len(G), "match": r3(np.mean([x["match"] for x in G])), "frac_ge": r3(np.mean([x["ge"] for x in G])), "jac5": r3(np.median([x["j5"] for x in G])),
             "jac1": r3(np.median([x["j1"] for x in G])), "n1_eq_tight": r3(np.mean([x["n1_eq"] for x in G])), "mean_d1": r3(np.mean([x["d1"] for x in G])), "mean_d5": r3(np.mean([x["d5"] for x in G]))}
out["E5"] = E5

# ---------------- E6
R6 = {r["run_id"]: r for r in runs("E6")}
p, h = 23, 11
def fold(k): k = k % p; return np.where(k > h, p - k, k)
pairs = []
for seed in range(6):
    a = R6[f"E6_pair_seed{seed}_u1"]
    wa = torch.load(os.path.join(a["dir"], "final.pt")); Ua, Wa = wa["U"].double(), wa["W"].double()
    Aa = np.asarray(a["m"]["A"])
    for u in (2, 5):
        b = R6[f"E6_pair_seed{seed}_u{u}"]
        wb = torch.load(os.path.join(b["dir"], "final.pt")); Ub, Wb = wb["U"].double(), wb["W"].double()
        Ab = np.asarray(b["m"]["A"]); j = np.arange(1, h + 1)
        mapped = Aa[fold(u * j) - 1]
        uinv = pow(u, -1, p); src = torch.tensor([(uinv * x) % p for x in range(p)])
        drift = float(torch.sqrt(((Ub - Ua[..., src]) ** 2).sum() + ((Wb - Wa[..., src]) ** 2).sum()) / torch.sqrt((Ua ** 2).sum() + (Wa ** 2).sum()))
        sa = set((np.flatnonzero(Aa >= .1 * Aa.max()) + 1).tolist()); sb = set((np.flatnonzero(Ab >= .1 * Ab.max()) + 1).tolist())
        sam = {jj for jj in range(1, h + 1) if int(fold(u * jj)) in sa}
        pairs.append({"seed": seed, "u": u, "st": (a["status"], b["status"]), "amp": float(np.abs(Ab - mapped).max() / Aa.max()), "drift": drift, "map": sb == sam,
                      "dg": abs(a["m"]["gamma"] - b["m"]["gamma"]) / abs(a["m"]["gamma"])})
E6 = {"pairs_map_exact": sum(x["map"] for x in pairs), "n_pairs": len(pairs), "all_ok": all(x["st"] == ("ok", "ok") for x in pairs),
      "amp_max": max(x["amp"] for x in pairs), "n_amp_below_7e-8": sum(x["amp"] < 7e-8 for x in pairs), "drift_median": float(np.median([x["drift"] for x in pairs])),
      "drift_max": max(x["drift"] for x in pairs), "dg_max": max(x["dg"] for x in pairs)}
for name, sel in (("ok", lambda r: r["status"] == "ok"), ("allpos", lambda r: r["m"]["margin"] > 0)):
    G = [r for k, r in R6.items() if k.startswith("E6_marg_") and r["m"] is not None and sel(r)]
    act = np.zeros(h); Ks = []
    for r in G:
        A = np.asarray(r["m"]["A"]); a_ = A >= .1 * A.max(); act += a_; Ks.append(int(a_.sum()))
    ch = stats.chisquare(act)
    E6[f"marg_{name}"] = {"n": len(G), "P": [round(v, 2) for v in act / len(G)], "EK_h": r3(np.mean(Ks) / h), "chi2": r3(ch.statistic), "p": r3(ch.pvalue),
                         "K_dist": dict(__import__("collections").Counter(Ks))}
out["E6"] = E6

# ---------------- E7
R7 = runs("E7"); idx7, P7 = preds("E7")
pl = P7["plateaus"]; p = 37; h = 18
def dil_all(y):
    j = np.arange(1, h + 1); D = []
    for u in range(1, h + 1):
        f = (u * j) % p; f = np.where(f > h, p - f, f); D.append(np.asarray(y)[f - 1])
    return np.array(D)
reps = [dil_all(x["rep"]) for x in pl]
def plateau_of(s):
    for k, x in enumerate(pl):
        if x["s_from"] - 1e-12 <= s <= x["s_to"] + 1e-12: return k
rows = []
for r in R7:
    m = r["m"]
    if m is None or m["margin"] <= 0: continue
    s = m["s"]; A = np.asarray(m["A"]); w = shares(A, s)
    tvs = [min(tv(w, shares(y, s)) for y in D) for D in reps]
    cell = idx7[key_of(r)]; Y = np.asarray(cell["y_all"]); y = Y[align(A, Y, s)]
    rows.append({"st": r["status"], "s": round(s, 4), "seed": r["c"]["seed"], "K": cell["K_star"], "thr10": thr(A, .1), "K2": energy_counts(A)["K_2pct"],
                 "pp": plateau_of(s), "pb": int(np.argmin(tvs)), "tv": tv(w, shares(y, s)), "cos": cos(bnorm(A, s), y), "eff": prog_value(A, p, s) / cell["Gamma"],
                 "gIF": m["gamma"] / (4 ** (-(4 - m["q"]) / m["q"]) * Bp(p, 4) * cell["Gamma"]), "gU": m["gamma"] / (Bp(p, 4) * cell["Gamma"]),
                 "sy": np.sort(np.asarray(m["yhat"]))[::-1][:20], "id": r["run_id"]})
E7 = {}
for g in ("ok", "not_converged"):
    G = [x for x in rows if x["st"] == g]
    within, across = [], []
    for a, b in itertools.combinations(sorted(G, key=lambda z: z["id"]), 2):
        d = 0.5 * np.abs(a["sy"] - b["sy"]).sum()
        if a["pp"] == b["pp"]: within.append(d)
        elif abs(a["pp"] - b["pp"]) == 1: across.append(d)
    E7[g] = {"n": len(G), "plateau_match": sum(x["pp"] == x["pb"] for x in G), "thr10": sweep([x["thr10"] for x in G], [x["K"] for x in G]),
             "K2": sweep([x["K2"] for x in G], [x["K"] for x in G]), "tv": r3(np.median([x["tv"] for x in G])), "cos": r3(np.median([x["cos"] for x in G])),
             "gIF": r3(np.median([x["gIF"] for x in G])), "gU_max": r3(max(x["gU"] for x in G)), "eff": r3(np.median([x["eff"] for x in G])),
             "within": r3(np.mean(within)), "adjacent": r3(np.mean(across)),
             "per": [(x["s"], x["seed"], x["K"], x["thr10"], x["K2"], x["pp"], x["pb"]) for x in sorted(G, key=lambda z: (z["s"], z["seed"]))]}
out["E7"] = E7

# ---------------- E8 (+ E1 nu=3)
idx8, _ = preds("E8")
rows = []
for e in ("E8", "E1"):
    for r in runs(e):
        m = r["m"]
        if m is None or r["c"]["n"] != 2 or r["status"] not in ("ok", "not_converged") or m["margin"] <= 0: continue
        k = key_of(r)
        if k not in idx8: continue
        cell = idx8[k]; A = np.asarray(m["A"]); s = m["s"]; Y = np.asarray(cell["y_all"]); y = Y[align(A, Y, s)]
        rows.append({"e": e, "st": r["status"], "p": m["p"], "s": round(s, 4), "K": cell["K_star"], "thr10": thr(A, .1), "K2": energy_counts(A)["K_2pct"],
                     "tv": tv(shares(A, s), shares(y, s)), "eff": prog_value(A, m["p"], s) / cell["Gamma"]})
E8 = {}
for s in (1.25, 1.5):
    G = [x for x in rows if x["s"] == s and x["st"] == "ok"]
    per = []
    for p in sorted({x["p"] for x in G}):
        gp = [x for x in G if x["p"] == p]
        per.append((p, gp[0]["K"], float(np.median([x["thr10"] for x in gp])), float(np.median([x["K2"] for x in gp])), len(gp), r3(np.median([x["tv"] for x in gp]))))
    E8[str(s)] = {"per_p": per, "p_match_thr10": sum(q[2] == q[1] for q in per), "n_p": len(per), "thr10": sweep([x["thr10"] for x in G], [x["K"] for x in G]),
                  "Klogp": [r3(min(q[1] / math.log(q[0]) for q in per)), r3(max(q[1] / math.log(q[0]) for q in per))],
                  "K2_cells_within1": sum(abs(q[3] - q[1]) <= 1 for q in per)}
E8["K2_cells_within1_total"] = E8["1.25"]["K2_cells_within1"] + E8["1.5"]["K2_cells_within1"]
E8["n_cells_total"] = E8["1.25"]["n_p"] + E8["1.5"]["n_p"]
E8["eff_E8_ok_median"] = r3(np.median([x["eff"] for x in rows if x["e"] == "E8" and x["st"] == "ok"]))
E8["eff_E8_ok_by_p"] = {p: r3(np.median([x["eff"] for x in rows if x["e"] == "E8" and x["st"] == "ok" and x["p"] == p])) for p in sorted({x["p"] for x in rows if x["e"] == "E8"})}
out["E8"] = E8

# ---------------- RESULTS cross-experiment: ceiling in every run, clean energy
ceil = []; clean = defaultdict(list)
for e in ("E1", "E3", "E6", "E7", "E8"):
    ix, _ = preds(e)
    for r in runs(e):
        m = r["m"]
        if m is None: continue
        c = r["c"]; k = key_of(r)
        if k not in ix: continue
        G = ix[k]["Gamma"]; fac = math.factorial(c["n"]) if c["model"] == "power" else 1
        if not np.isfinite(m["gamma"]): continue
        ceil.append((e, r["run_id"], r["status"], m["gamma"] / (fac * Bp(c["p"], c["n"] + 1) * G)))
        if r["status"] == "ok": clean[e].append(m["energy_frac"]["clean"])
out["ceiling"] = {"n_runs": len(ceil), "max": max(x[3] for x in ceil), "argmax": max(ceil, key=lambda x: x[3])[:3],
                  "max_by_exp": {e: r3(max(x[3] for x in ceil if x[0] == e)) for e in ("E1", "E3", "E6", "E7", "E8")}}
allc = np.concatenate([clean[e] for e in ("E1", "E6", "E7", "E8")])
out["clean_energy_ok"] = {e: [r3(min(v)), r3(np.median(v)), r3(max(v))] for e, v in clean.items()}
out["clean_energy_ok_A1exps_pooled"] = [r3(allc.min()), r3(np.median(allc)), r3(allc.max())]
json.dump(out, open(os.path.join(ROOT, "audit", "recompute_rest.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
for k, v in out.items():
    print("==", k); print(json.dumps(v, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))[:3000])
