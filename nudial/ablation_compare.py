"""Compare the interface-ablation count (nudial/ablation_count.py) with the
conventional DFT-sort-and-cut instruments on the same runs.

Per cohort: count histograms, exact / within-1 agreement of K_abl with
K_thr10, K_gap, K_load90; SET agreement (Jaccard of the ablation-counted
frequency set vs the legacy top-K_thr10 set by clean amplitude); and the
list of disagreeing runs with both profiles, so every mismatch is inspectable.

Usage: uv run python nudial/ablation_compare.py notes/ablation_count_*.json
"""
import collections, json, sys
import numpy as np

rows = []
for f in sys.argv[1:]:
    rows += json.load(open(f))


EPS = 1e-6
def recount(r, fold_n=5, fold_e=0.25, stat="d_min", safety=1.0):
    """Final rule from stored per-k data: significant (both-cut > null floor),
    read-and-write (in/out cuts > their own null floors), harmonic fold in
    interface-energy order. Returns (counted, raw, rw, classes)."""
    p = r["p"]; per = {int(k): v for k, v in r["per"].items()}; null = r["null"]; cand = r["cand"]
    fl = safety * max([per[k]["both"][stat] for k in null] + [EPS])
    raw = [k for k in cand if per[k]["both"][stat] > fl]
    e = {k: per[k]["e_in"] + per[k]["e_out"] for k in cand}
    if "in" in per[cand[0]]:
        fi = safety * max([per[k]["in"][stat] for k in null] + [EPS]); fo = safety * max([per[k]["out"][stat] for k in null] + [EPS])
        is_rw = lambda k: per[k]["in"][stat] > fi and per[k]["out"][stat] > fo
    else:
        is_rw = lambda k: True
    def harm(k, fnds):
        for f1 in fnds:
            if any((n * f1) % p in (k, p - k) for n in range(2, fold_n + 1)): return f1
        for i, f1 in enumerate(fnds):
            for f2 in fnds[i + 1:]:
                for n1 in range(-2, 3):
                    for n2 in range(-2, 3):
                        if n1 and n2 and (n1 * f1 + n2 * f2) % p in (k, p - k): return f1 if e[f1] >= e[f2] else f2
        return None
    counted, classes = [], {}
    for k in sorted(raw, key=lambda k: -e[k]):
        o = harm(k, counted)
        if o is not None and e[k] < fold_e * e[o]: classes[o].append(k)
        elif not is_rw(k): pass
        else: counted.append(k); classes[k] = []
    return counted, raw, [k for k in raw if is_rw(k)], classes

import os
STATS = os.environ.get("ABL_STATS", "d_min,d_q01,d_mean").split(",")
SAFETY = float(os.environ.get("ABL_SAFETY", "1.0"))
for STAT in STATS:
    print(f"\n########## stat = {STAT}   (null-floor safety factor {SAFETY})")
    for r in rows:
        c, raw, rw, cl = recount(r, stat=STAT, safety=SAFETY)
        r["K_abl"], r["counted"], r["K_abl_raw"], r["K_rw"], r["classes"] = len(c), c, len(raw), len(rw), {str(k): v for k, v in cl.items()}

    def jac(a, b):
        a, b = set(a), set(b)
        return len(a & b) / len(a | b) if (a | b) else 1.0

    print(f"{'cohort':52s} {'n':>3s} | {'K_abl':>14s} | {'K_thr10':>14s} | {'=thr10':>6s} {'±1':>5s} | {'=gap':>5s} | {'=ld90':>5s} | {'setJ':>5s} | {'raw-abl':>7s} | {'floor':>6s}")
    mism = []
    for cell in dict.fromkeys(r["cell"] for r in rows):
        rs = [r for r in rows if r["cell"] == cell]
        h = lambda f: dict(sorted(collections.Counter(f(r) for r in rs).items(), key=lambda kv: str(kv[0])))
        L = lambda r: r["legacy"]
        if "K_true" in rs[0]:
            ex = np.mean([r["K_abl"] == r["K_true"] for r in rs]); ex_leg = np.mean([L(r)["K_thr10"] == r["K_true"] for r in rs])
            setj = np.mean([jac(r["counted"], r["S"]) for r in rs])
            print(f"{cell:52s} {len(rs):3d} | {str(h(lambda r: r['K_abl'])):>14s} | {str(h(lambda r: L(r)['K_thr10'])):>14s} | vs TRUE: abl exact {ex:.2f}, thr10 exact {ex_leg:.2f}, set J {setj:.2f}")
            continue
        e10 = np.mean([r["K_abl"] == L(r)["K_thr10"] for r in rs]); w1 = np.mean([abs(r["K_abl"] - L(r)["K_thr10"]) <= 1 for r in rs])
        eg = np.mean([r["K_abl"] == L(r)["K_gap"] for r in rs]); el = np.mean([r["K_abl"] == L(r)["K_load90"] for r in rs])
        setj = np.mean([jac(r["counted"], L(r)["top_k"][:L(r)["K_thr10"]]) for r in rs])
        rawd = np.mean([r["K_abl_raw"] - r["K_abl"] for r in rs])
        print(f"{cell:52s} {len(rs):3d} | {str(h(lambda r: r['K_abl'])):>14s} | {str(h(lambda r: L(r)['K_thr10'])):>14s} | {e10:6.2f} {w1:5.2f} | {eg:5.2f} | {el:5.2f} | {setj:5.2f} | {rawd:7.2f} | {np.median([r['null_floor'] for r in rs]):6.4f}")
        mism += [r for r in rs if r["K_abl"] != L(r)["K_thr10"]]

    print(f"\nDISAGREEMENTS (K_abl != K_thr10): {len(mism)}")
    for r in mism:
        L = r["legacy"]
        print(f"  {r['name']}: K_abl {r['K_abl']} (raw {r['K_abl_raw']}, rw {r['K_rw']}) vs thr10 {L['K_thr10']} gap {L['K_gap']} ld90 {L['K_load90']} | floor {r['null_floor']:.4f}")
        print(f"     ablation profile {[(k, d) for k, d in r['profile'][:9]]}")
        print(f"     legacy top_k     {list(zip(L['top_k'], L['top8']))}")
        if any(r["classes"].values()):
            print(f"     folded           {dict((k, v) for k, v in r['classes'].items() if v)}")
