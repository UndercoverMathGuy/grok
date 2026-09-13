"""Z programme tables (PLAN_Z.md): per cell K_abl mode vs K*, gamma_2 vs the constructed ceiling, coverage-optimality of
each counted committee M(S_run) / M_K(|S_run|), non-faithful membership (Z_120), Z_11^2 nu-6 mode; ZL ladder best per support.
Usage: uv run python heracles/analyze_Z.py --cohort Z1 | ZR | ZL"""
import argparse, collections, json, math, re, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from groups import get_group
KS = json.load(open(HERE / "kstar_ab.json"))


def load(cohort):
    rows = []
    for f in sorted((HERE / "scores" / cohort).glob("*.json")):
        if "@ckpt" in f.stem: continue
        for r in json.load(open(f))["rows"]:
            r["cell"] = f.stem; rows.append(r)
    return rows


def freq(G, k):
    """frequency index of a Z_120 irrep name: fNN -> NN, sgn60 -> 60."""
    n = G.irrep_names[k]; return int(n[1:]) if n.startswith("f") else int(n[3:])


def cov_frac(G, S):
    """M(S) / M_K(|S|): coverage of the counted set relative to the ladder optimum at that size."""
    if not S: return float("nan")
    cov = G.coverage(); x = [i for i in range(G.N) if i != G.e]
    M = cov[np.ix_(S, x)].sum(0).min(); MK = KS[G.name]["ladder"].get(str(len(S)), {}).get("M")
    return M / MK if MK else float("nan")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cohort", default="Z1"); a = ap.parse_args()
    rows = load(a.cohort); cells = collections.OrderedDict()
    for r in rows: cells.setdefault(r["cell"], []).append(r)
    for c, rs in cells.items():
        task = c.split("_")[0]; G = get_group(task); nu = int(re.search(r"_nu(\d)", c).group(1)) if "_nu" in c else 4
        g = [r for r in rs if r.get("acc", 0) >= 0.99 and r.get("K_abl") is not None]
        K = collections.Counter(r["K_abl"] for r in g); Ks = KS[task][f"kstar_nu{nu}"]; plat = KS[task][f"plateau74_nu{nu}"]
        ceil = KS[task].get(f"ceiling_nu{nu}")
        print(f"\n{c}: grok {len(g)}/{len(rs)}  K* {Ks} plateau {plat}")
        if not g: continue
        mode = max(K, key=lambda k: (K[k], -k)); g2 = float(np.median([r["gamma2"] for r in g]))
        cf = [cov_frac(G, r["counted"]) for r in g]
        nonf = [any(not G.faithful[k] or math.gcd(freq(G, k), 120) > 1 for k in r["counted"]) if task == "z120" else None for r in g]
        gcds = collections.Counter(math.gcd(freq(G, k), 120) for r in g for k in r["counted"]) if task == "z120" else None
        print(f"  K_abl {dict(sorted(K.items()))}  mode {mode}  (a) mode in {{K*, K*+1}}: {mode in (Ks, Ks + 1)}  in plateau: {sum(K[k] for k in K if k in plat) / len(g):.2f}")
        print(f"  (b) gamma_2 med {g2:.3g} / ceiling {ceil:.3g} = {g2 / ceil:.2f}   max {max(r['gamma2'] for r in g) / ceil:.2f}   clean med {np.median([r['legacy']['clean_frac'] for r in g]):.2f}")
        print(f"  (c) coverage-optimality M(S)/M_K: med {np.nanmedian(cf):.2f}  min {np.nanmin(cf):.2f}  frac >= 0.74: {np.mean([x >= 0.74 for x in cf]):.2f}")
        if task == "z120":
            print(f"  (d) committees with a non-faithful (gcd > 1) member: {np.mean(nonf):.2f}   gcd histogram of counted frequencies {dict(sorted(gcds.items()))}")
        if task == "z11x11":
            dirs = [[tuple(int(t) for t in G.irrep_names[k][1:].split("+")) for k in r["counted"]] for r in g]
            par = lambda a, b: (a[0] * b[1] - a[1] * b[0]) % 11 == 0
            npar = [sum(par(d[i], d[j]) for i in range(len(d)) for j in range(i + 1, len(d))) for d in dirs]
            print(f"  (e) parallel pairs inside counted sets: {sum(npar)} of {sum(len(d) * (len(d) - 1) // 2 for d in dirs)}   nu-6 statement: mode {mode}" + ("" if nu != 6 else f" (kill if 3 with < 25% at 4: {K.get(4, 0) / len(g):.2f} at 4)"))
        ex = sorted(g, key=lambda r: -r["gamma2"])[:3]
        print("  top runs:", [(r["K_abl"], [G.irrep_names[k] for k in r["counted"]], f"{r['gamma2']:.2e}") for r in ex])


if __name__ == "__main__":
    main()
