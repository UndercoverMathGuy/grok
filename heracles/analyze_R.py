"""Cohort R recipe table + selection (PLAN.md rule): per cell grok rate (acc >= 0.99), finite, median
gamma_2 / gamma_q, clean fraction, count histograms. Per optimizer class pick: >= 7/8 grokked, all finite,
then highest median gamma (gamma_2 for q = 2 classes, gamma_q otherwise), ties -> smaller wd.

Usage: uv run python heracles/analyze_R.py [--scores heracles/scores/R] [--min-grok 7]
"""
import argparse, collections, json, math, re
from pathlib import Path

import numpy as np

CEIL_G2 = 6.39e-6     # constructed K=4 free-amplitude ceiling, relu-4 m512 p113 (notes/construct_gamma.json)


def cls_of(cell):
    if cell.startswith("gd_"): return "gd"
    if cell.startswith("gdm_"): return "gdm"
    if cell.startswith("sgd_"): return "sgd"
    if cell.startswith("lq_q3"): return "lq3"
    if cell.startswith("lq_q4"): return "lq4"
    if "MIXED" in cell: return "mixed"
    if cell.startswith("signm_"): return "signm"
    if cell.startswith("sign_"): return "sign"
    return "?"


def wd_of(cell):
    m = re.search(r"_(wd|box)([0-9.e-]+)", cell)
    return float(m.group(2)) if m else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(Path(__file__).resolve().parent / "scores" / "R"))
    ap.add_argument("--min-grok", type=int, default=7)
    args = ap.parse_args()
    rows = []
    for f in sorted(Path(args.scores).glob("*.json")):
        d = json.load(open(f)); rs = d["rows"]
        cell = f.stem; g = [r for r in rs if r.get("acc", 0) >= 0.99 and r.get("K_abl") is not None]
        fin = [r for r in rs if math.isfinite(r.get("m_min", float("nan")))]
        med = lambda k, src=g: float(np.median([r[k] for r in src])) if src else float("nan")
        leg = lambda k: dict(sorted(collections.Counter(r["legacy"][k] for r in g).items()))
        rows.append(dict(cell=cell, cls=cls_of(cell), wd=wd_of(cell), n=len(rs), grok=len(g), finite=len(fin),
                         g2=med("gamma2"), gq=med("gammaq"), clean=med("clean_frac") if False else (float(np.median([r["legacy"]["clean_frac"] for r in g])) if g else float("nan")),
                         mmin=med("m_min"), K_abl=dict(sorted(collections.Counter(r["K_abl"] for r in g).items())), K_thr10=leg("K_thr10"),
                         acc_med=float(np.median([r.get("acc", 0) for r in rs]))))
    print(f"{'cell':44s} {'n':>2} {'grok':>4} {'fin':>3} {'g2 med':>9} {'g2/ceil':>7} {'gq med':>9} {'clean':>5} {'m_min':>6}  K_abl | K_thr10")
    for cls in ("gd", "gdm", "sgd", "lq3", "lq4", "sign", "signm", "mixed"):
        sub = [r for r in rows if r["cls"] == cls]
        if not sub: continue
        print(f"--- {cls}")
        for r in sorted(sub, key=lambda r: r["cell"]):
            print(f"{r['cell']:44s} {r['n']:2d} {r['grok']:4d} {r['finite']:3d} {r['g2']:9.3g} {r['g2']/CEIL_G2:7.2f} {r['gq']:9.3g} {r['clean']:5.2f} {r['mmin']:6.2f}  {r['K_abl']} | {r['K_thr10']}")
    print("\n==== selection (>= %d/8 grokked, all finite, max median gamma, ties -> smaller wd)" % args.min_grok)
    for cls in ("gd", "gdm", "sgd", "lq3", "lq4", "sign", "signm", "mixed"):
        ok = [r for r in rows if r["cls"] == cls and r["grok"] >= args.min_grok and r["finite"] == r["n"]]
        key = (lambda r: r["g2"]) if cls in ("gd", "gdm", "sgd") else (lambda r: r["gq"])
        if not ok:
            best = max([r for r in rows if r["cls"] == cls], key=lambda r: (r["grok"], key(r)), default=None)
            print(f"{cls:6s}: NO cell meets the bar; best grok rate = {best['cell'] if best else '-'} ({best['grok'] if best else 0}/8)")
            continue
        ok.sort(key=lambda r: (-round(key(r), 12), r["wd"]))
        b = ok[0]
        print(f"{cls:6s}: {b['cell']}  grok {b['grok']}/8  gamma2 {b['g2']:.3g} ({b['g2']/CEIL_G2:.2f} of ceiling)  gammaq {b['gq']:.3g}  clean {b['clean']:.2f}  K_abl {b['K_abl']}  [{len(ok)} cells qualify]")


if __name__ == "__main__":
    main()
