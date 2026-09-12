"""H3 table: per (nu, wd, budget) the count histogram, mode, gamma_2 / ceiling, clean fraction, train CE.
Usage: uv run python heracles/analyze_H3.py [--scores heracles/scores/H3]"""
import argparse, collections, json, re
from pathlib import Path
import numpy as np
CEIL = {4: 6.39e-6}     # constructed K=4 ceiling, relu-4 m512 p113 (no nu-6 ceiling yet: H6)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scores", default=str(Path(__file__).resolve().parent / "scores" / "H3"))
    a = ap.parse_args(); rows = []
    for f in Path(a.scores).glob("*.json"):
        d = json.load(open(f))["rows"]; nu = d[0]["nu"]
        wd = float(re.search(r"_wd([0-9.e-]+)_", f.stem).group(1)); m = re.search(r"@ckpt_(\d+)", f.stem)
        budget = int(m.group(1)) if m else (600_000 if nu == 4 else 200_000)
        g = [r for r in d if r.get("acc", 0) >= 0.99 and r.get("K_abl") is not None]
        K = collections.Counter(r["K_abl"] for r in g)
        rows.append(dict(nu=nu, wd=wd, budget=budget, n=len(d), grok=len(g), K=dict(sorted(K.items())),
                         mode=max(K, key=lambda k: (K[k], -k)) if K else None,
                         g2=float(np.median([r["gamma2"] for r in g])) if g else float("nan"),
                         clean=float(np.median([r["legacy"]["clean_frac"] for r in g])) if g else float("nan"),
                         mmin=float(np.median([r["m_min"] for r in g])) if g else float("nan")))
    print(f"{'nu':>2} {'wd':>7} {'budget':>7} {'grok':>5} {'mode':>4} {'g2':>9} {'/ceil':>6} {'clean':>5} {'m_min':>6}  hist")
    for r in sorted(rows, key=lambda r: (r["nu"], -r["wd"], r["budget"])):
        ceil = CEIL.get(r["nu"]); print(f"{r['nu']:>2} {r['wd']:>7g} {r['budget']:>7d} {r['grok']:>2}/{r['n']:<2} {r['mode']!s:>4} {r['g2']:9.2e} {(r['g2']/ceil if ceil else float('nan')):6.2f} {r['clean']:5.2f} {r['mmin']:6.2f}  {r['K']}")

if __name__ == "__main__":
    main()
