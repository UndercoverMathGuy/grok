"""S5 programme tables (PLAN_S5.md). Reads heracles/scores/<cohort>/*.json.
  --cohort SR : E1 recipe table (grok, median gamma_2, leader histogram, sign_rel) -> RECIPES rule winner
  --cohort S0 : E0 measured ladder: per (nu, support) best + median gamma_2 over seeds, predicted leader per nu
  --cohort S1 : E2 statements (a)-(d) per cell against the S0 ladder (pass --ladder heracles/scores/S0)
  --cohort S3 : same tables for s4 / a5 / d113 (ladder cells = names with sup-)
Usage: uv run python heracles/analyze_S5.py --cohort SR"""
import argparse, collections, json, re
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def load(cohort):
    rows = []
    for f in sorted((HERE / "scores" / cohort).glob("*.json")):
        d = json.load(open(f))["rows"]
        for r in d:
            r["cell"] = f.stem; rows.append(r)
    return rows


def grokked(rs):
    return [r for r in rs if r.get("acc", 0) >= 0.99 and r.get("K_abl") is not None]


def cells(rows):
    out = collections.OrderedDict()
    for r in rows: out.setdefault(r["cell"], []).append(r)
    return out


def med(xs):
    return float(np.median(xs)) if len(xs) else float("nan")


def recipe_table(rows):
    print(f"{'cell':60} {'grok':>5} {'g2 med':>9} {'g2 max':>9} {'clean':>5} {'K_abl':>12} leader (share med, sign_rel med)")
    for c, rs in cells(rows).items():
        g = grokked(rs); K = collections.Counter(r["K_abl"] for r in g); L = collections.Counter(r.get("leader") for r in g)
        print(f"{c:60} {len(g):>2}/{len(rs):<2} {med([r['gamma2'] for r in g]):9.3g} {max([r['gamma2'] for r in g], default=float('nan')):9.3g} "
              f"{med([r['legacy']['clean_frac'] for r in g]):5.2f} {str(dict(sorted(K.items()))):>12} {dict(L)} "
              f"({med([r['lead_share'] for r in g]):.2f}, {med([r['sign_rel'] for r in g if r.get('sign_rel') is not None]):.3f})")


def ladder(rows):
    """per (task, nu): support -> best / median gamma_2 over seeds (grokked). Ceiling = best."""
    out = {}
    for c, rs in cells(rows).items():
        if "sup-" not in c: continue
        m = re.search(r"^(\w+?)_.*sup-([\w-]+?)_lr", c); task, sup = m.group(1), m.group(2)
        nu = int(re.search(r"_nu(\d)", c).group(1)) if "_nu" in c else 4
        g = grokked(rs)
        out.setdefault((task, nu), {})[sup] = dict(n=len(rs), grok=len(g), best=max([r["gamma2"] for r in g], default=float("nan")),
                                                     med=med([r["gamma2"] for r in g]), m_min=med([r["m_min"] for r in g]),
                                                     sign_rel=med([r["sign_rel"] for r in g if r.get("sign_rel") is not None]))
    for (task, nu), sups in sorted(out.items()):
        print(f"\n== {task} nu {nu}: measured ladder (fixed-support gd, gamma_2 best / median over seeds)")
        best = max(sups, key=lambda s: sups[s]["best"] if np.isfinite(sups[s]["best"]) else -1)
        for s, v in sorted(sups.items(), key=lambda kv: -kv[1]["best"] if np.isfinite(kv[1]["best"]) else 0):
            print(f"  {s:18} grok {v['grok']}/{v['n']}  best {v['best']:9.3g}  med {v['med']:9.3g}  m_min {v['m_min']:6.3f}  sign_rel {v['sign_rel']:.3f}  {'<- argmax' if s == best else ''}")
    return out


def statements(rows, lad):
    for c, rs in cells(rows).items():
        if "sup-" in c: continue
        task = c.split("_")[0]; nu = int(re.search(r"_nu(\d)", c).group(1)) if "_nu" in c else 4
        g = grokked(rs); n = len(g)
        if not n: print(f"{c}: 0 grokked"); continue
        L = collections.Counter(r["leader"] for r in g)
        sups = lad.get((task, nu), {})
        pred = max(sups, key=lambda s: sups[s]["best"]) if sups else None
        pred_leader = pred.split("-")[0] if pred else None
        a = sum(1 for r in g if r["lead_share"] >= 0.9) / n
        a_kill = sum(1 for r in g if r["n_faithful_02"] >= 2)
        b = L.get(pred_leader, 0) / n if pred_leader else float("nan")
        sr = [r["sign_rel"] for r in g if r.get("sign_rel") is not None]
        std_sr = [r["sign_rel"] for r in g if r["leader"] == "std" and r.get("sign_rel") is not None]
        oth_sr = [r["sign_rel"] for r in g if r["leader"] != "std" and r.get("sign_rel") is not None]
        g2 = med([r["gamma2"] for r in g])
        ceil = max((v["best"] for s, v in sups.items()), default=float("nan"))
        print(f"\n{c}: grok {n}/{len(rs)}  leaders {dict(L)}  K_abl {dict(sorted(collections.Counter(r['K_abl'] for r in g).items()))}")
        print(f"  (a) one faithful irrep >= 0.9 share: {a:.2f} of runs (kill if >= 2 faithful above 0.2 in >= 3: {a_kill})")
        print(f"  (b) predicted leader {pred_leader} (support {pred}) leads {b:.2f} of runs")
        print(f"  (c) sign_rel: std-led med {med(std_sr):.3f} [{min(std_sr, default=float('nan')):.3f}, {max(std_sr, default=float('nan')):.3f}]  other-led med {med(oth_sr):.3f}")
        print(f"  (d) gamma_2 med {g2:.3g} / ladder ceiling {ceil:.3g} = {g2 / ceil if ceil == ceil else float('nan'):.2f}   clean med {med([r['legacy']['clean_frac'] for r in g]):.2f}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cohort", required=True); ap.add_argument("--ladder", default=None)
    a = ap.parse_args(); rows = load(a.cohort)
    if a.cohort == "SR":
        recipe_table(rows)
    elif a.cohort == "S0":
        ladder(rows)
    else:
        lad = ladder(load(a.ladder) if a.ladder else rows) if (a.ladder or a.cohort == "S3") else {}
        recipe_table(rows); statements(rows, lad)


if __name__ == "__main__":
    main()
