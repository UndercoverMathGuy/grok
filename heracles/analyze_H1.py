"""H1 table: per (p, nu) the registered primary (lr 3) and fallback (lr 1) cells vs K*(p, nu, q=2) and plateau74.
Usage: uv run python heracles/analyze_H1.py [--scores heracles/scores/H1]"""
import argparse, collections, json, re
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scores", default=str(Path(__file__).resolve().parent / "scores" / "H1"))
    a = ap.parse_args(); ks = json.load(open(Path(__file__).resolve().parent / "kstar.json"))
    cells = {}
    for f in sorted(Path(a.scores).glob("*.json")):
        rows = json.load(open(f))["rows"]; p, nu = rows[0]["p"], rows[0]["nu"]
        lr = float(re.search(r"lr([0-9.]+)", f.stem).group(1))
        g = [r for r in rows if r.get("acc", 0) >= 0.99 and r.get("K_abl") is not None]
        K = collections.Counter(r["K_abl"] for r in g)
        cells[(p, nu, lr)] = dict(n=len(rows), grok=len(g), K=dict(sorted(K.items())),
                                 mode=max(K, key=lambda k: (K[k], -k)) if K else None,
                                 g2=float(np.median([r["gamma2"] for r in g])) if g else float("nan"),
                                 clean=float(np.median([r["legacy"]["clean_frac"] for r in g])) if g else float("nan"), Ks=[r["K_abl"] for r in g])
    print(f"{'p':>4} {'nu':>2} {'K*':>3} {'plateau':>8} | {'lr3 grok':>8} {'mode':>4} {'in-plat':>7} {'g2':>9} {'clean':>5}  hist | lr1 mode hist")
    for p in (59, 71, 97, 113, 127):
        for nu in (3, 4, 5, 6):
            e = ks[str(p)]["pred"][f"nu{nu}_q2"]
            kstar, plat = e["K_star"], (min(e["plateau74"]), max(e["plateau74"]))
            c3, c1 = cells.get((p, nu, 3.0)), cells.get((p, nu, 1.0))
            if c3:
                inp = np.mean([plat[0] <= k <= plat[1] for k in c3["Ks"]]) if c3["Ks"] else float("nan")
                s3 = f"{c3['grok']:>5}/{c3['n']:<2} {c3['mode']!s:>4} {inp:7.2f} {c3['g2']:9.2e} {c3['clean']:5.2f}  {c3['K']}"
            else: s3 = "   (unscored)"
            s1 = f"{c1['mode']} {c1['K']}" if c1 else "(unscored)"
            print(f"{p:>4} {nu:>2} {kstar:>3} {f"{plat[0]}..{plat[1]}":>8} | {s3} | {s1}")

if __name__ == "__main__":
    main()
