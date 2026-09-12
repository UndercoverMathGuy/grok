"""H4 geometry dial: per (nu, q) grid, every cell + the RECIPES-rule winner (>= 7/8 grok, finite, max median gamma_q,
ties -> smaller wd). Usage: uv run python heracles/analyze_H4.py [--scores heracles/scores/H4]"""
import argparse, collections, json, math, re
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scores", default=str(Path(__file__).resolve().parent / "scores" / "H4"))
    a = ap.parse_args(); ks = json.load(open(Path(__file__).resolve().parent / "kstar.json"))["113"]["pred"]; cells = []
    for f in sorted(Path(a.scores).glob("*.json")):
        d = json.load(open(f))["rows"]
        if not d: continue
        opt = f.stem.split("_")[0]; nu = int(re.search(r"_nu(\d)", f.stem).group(1)) if "_nu" in f.stem else 4
        mq = re.search(r"_q(\d)", f.stem); q = float(mq.group(1)) if mq else float("inf")
        lr = float(re.search(r"lr([0-9.e-]+)", f.stem).group(1)); m = re.search(r"(wd|box)([0-9.e-]+)", f.stem); wd = float(m.group(2))
        g = [r for r in d if r.get("acc", 0) >= 0.99 and r.get("K_abl") is not None]; fin = all(math.isfinite(r.get("m_min", float("nan"))) for r in d)
        K = collections.Counter(r["K_abl"] for r in g)
        cells.append(dict(nu=nu, q=q, opt=opt, lr=lr, wd=wd, n=len(d), grok=len(g), fin=fin, K=dict(sorted(K.items())),
                          mode=max(K, key=lambda k: (K[k], -k)) if K else None,
                          gq=float(np.median([r["gammaq"] for r in g])) if g else float("nan"),
                          g2=float(np.median([r["gamma2"] for r in g])) if g else float("nan"),
                          clean=float(np.median([r["legacy"]["clean_frac"] for r in g])) if g else float("nan"),
                          ce=float(np.median([r.get("train_ce", float("nan")) for r in g])) if g else float("nan")))
    for (nu, q, opt) in sorted({(c["nu"], c["q"], c["opt"]) for c in cells}, key=lambda t: (t[0], t[1] if math.isfinite(t[1]) else 99)):
        qs = "inf" if not math.isfinite(q) else f"{q:g}"; pred = ks.get(f"nu{nu}_q{qs}", {})
        print(f"--- nu {nu}  q {qs} ({opt})   predicted K* {pred.get('K_star')}  plateau74 {pred.get('plateau74')}  K*_or_dense {pred.get('K_star_or_dense')}")
        sub = sorted([c for c in cells if (c["nu"], c["q"], c["opt"]) == (nu, q, opt)], key=lambda c: (c["lr"], c["wd"]))
        for c in sub:
            print(f"   lr {c['lr']:<6g} wd/box {c['wd']:<7g} grok {c['grok']}/{c['n']} {'fin' if c['fin'] else 'NONFIN'}  mode {c['mode']!s:>4}  gq {c['gq']:9.3g}  g2 {c['g2']:9.2e}  clean {c['clean']:.2f}  {c['K']}")
        ok = [c for c in sub if c["grok"] >= 7 and c["fin"]]
        if ok:
            b = sorted(ok, key=lambda c: (-round(c["gq"], 12), c["wd"]))[0]
            print(f"   WINNER lr {b['lr']:g} wd/box {b['wd']:g}: mode {b['mode']}  {b['K']}  gq {b['gq']:.3g}  [{len(ok)} qualify]")
        else: print("   no cell meets the bar")

if __name__ == "__main__":
    main()
