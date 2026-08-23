"""S-side content dissociation — the PRE-REGISTERED analysis.

Committed before Phase C trains. Runs unchanged on Phase B as a
retrospective (`--manifest .../phaseB_manifest.json`), which is the point:
the same code produces both numbers, so the reader can see what the design
fix bought.

The question: once the STRENGTH of the installed tilt and the base init are
controlled, does the CONTENT of what was installed move learning time?

Phase B could not answer it. Its 96 target sets were all distinct (a fresh
draw inside the innermost loop), so the "identity" factor was an index
label carrying no content and its within-dose permutation was vacuous by
construction. Phase C crosses ONE shared content pool with dose and base,
which makes the identity contrast (T1) real. T2-T4 are content coordinates
that vary in both phases and are therefore comparable across them.

Tests, all permuting the dose x base-residualised t_gen WITHIN its cell:

  T1  content identity (categorical; the crossed design's primary test)
  T2  content coordinates: LP margin pctile / relative margin, mean target
      frequency, target span, lowest target, max T_k rank
  T3  Mantel — do runs with more similar content have more similar t_gen
  T4  realisation: did the install take (exact set), and J(realised,
      installed) — learning should not care whether it took either

Scoring is by EQUIVALENCE (TOST), not by a non-significant p-value. The
band is declared here, before the data: a content effect is judged absent
if the 90% CI on the content-attributable spread in t_gen lies entirely
below BAND * median(t_gen). Absence-of-evidence p-values are reported
alongside but are NOT the verdict.

Usage:
    uv run python compiler/analyze_sside.py                      # Phase C
    uv run python compiler/analyze_sside.py --manifest \
        compiler/arms/phaseB_manifest.json --out notes/sside_phaseB.json
"""

import argparse
import collections
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dial"))
from analyze_dial import committee_from_coeffs, jaccard  # noqa: E402

BAND = 0.15          # pre-registered equivalence band, fraction of median t_gen
N_PERM = 20_000
ALPHA = 0.05


def load_rows(manifest):
    """One row per arm that reached the grokking threshold twice running."""
    rows = []
    for a in json.load(open(manifest)):
        rd = Path(a["run_dir"])
        f = rd / "spectra.npz"
        if not f.exists():
            continue
        z = np.load(f)
        ep, acc = np.asarray(z["epochs"]), np.asarray(z["test_acc"])
        ok = acc >= 0.99
        h = np.where(ok[:-1] & ok[1:])[0]
        if not len(h):
            continue
        S = tuple(a["targets"])
        com = tuple(sorted(committee_from_coeffs(z["coeffs"][-1])))
        cell, rep, fe = a["cell"], a["report"], a.get("feasibility")
        rows.append(dict(
            tag=a["tag"], base=cell.get("base") or a["base"], s=float(cell["s"]),
            content=cell.get("content"), K=cell.get("k"), S=S, com=com,
            t=float(ep[h[0]]), grid=int(np.diff(ep)[0]), is_ctrl=not S,
            lp_pct=(fe or {}).get("lp_pct"), lp_relM=(fe or {}).get("lp_relM"),
            meanf=float(np.mean(S)) if S else None,
            spanf=float(max(S) - min(S)) if S else None,
            minf=float(min(S)) if S else None,
            rankmax=float(max(rep["target_ranks"].values())) if S else None,
            escale=(max(e["energy_scale"] for e in rep["edits"].values())
                    if S else None),
            exact=(com == S) if S else None,
            J=jaccard(set(com), set(S)) if S else None))
    return rows


def residualise(rows, y, keys):
    """t_gen with the mean of its (dose, base, ...) cell removed."""
    cells = collections.defaultdict(list)
    for i, r in enumerate(rows):
        cells[tuple(r[k] for k in keys)].append(i)
    e = np.zeros(len(rows))
    for idx in cells.values():
        idx = np.array(idx)
        e[idx] = y[idx] - y[idx].mean()
    return e, cells


def perm_test(e, cells, stat, rng, n=N_PERM):
    obs = stat(e)
    null = np.empty(n)
    for b in range(n):
        v = np.empty_like(e)
        for idx in cells.values():
            idx = np.array(idx)
            v[idx] = e[idx][rng.permutation(len(idx))]
        null[b] = stat(v)
    return float(obs), float((null >= obs).mean()), null


# Every statistic below returns EPOCHS of t_gen attributable to the content
# coordinate, so the observed value, the permutation null and the
# equivalence band are all on one scale and directly comparable. Each is a
# linear functional of the residual vector, so permuting it is valid.

def cat_stat(labels):
    """RMS of the per-level mean residual: the typical t_gen offset a
    content level carries."""
    lv = np.unique(labels)
    return lambda v: float(np.sqrt(np.mean(
        [v[labels == l].mean() ** 2 for l in lv])))


def num_stat(x):
    """t_gen swing predicted across the full observed range of x."""
    xc = np.asarray(x, float) - np.mean(x)
    var, rng_x = float((xc ** 2).sum()), float(np.ptp(x))
    return lambda v: abs(float(xc @ v) / var) * rng_x


def tost(obs, null, med_t, band):
    """Equivalence verdict: is the content-attributable spread credibly
    below band*median(t_gen)? The permutation null gives the sampling
    distribution of the statistic under NO content effect; its upper 95th
    percentile is the largest spread this design would routinely produce
    from noise alone, so an observed spread below it AND below the band is
    equivalence, an observed spread above the band's noise-corrected
    ceiling is a detected effect."""
    ceiling = float(np.quantile(null, 1 - ALPHA))
    limit = band * med_t
    return dict(observed=obs, noise_ceiling=ceiling, band_epochs=limit,
                equivalent=bool(obs <= ceiling and obs <= limit),
                detected=bool(obs > ceiling and obs > limit))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",
                    default=str(ROOT / "compiler/arms/phaseC_manifest.json"))
    ap.add_argument("--out", default=str(ROOT / "notes/sside_summary.json"))
    ap.add_argument("--band", type=float, default=BAND)
    ap.add_argument("--seed", type=int, default=20260816)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    rows = load_rows(args.manifest)
    ctrl = [r for r in rows if r["is_ctrl"]]
    rows = [r for r in rows if not r["is_ctrl"]]
    if not rows:
        sys.exit(f"no completed arms under {args.manifest}")
    y = np.array([r["t"] for r in rows])
    med = float(np.median(y))
    grid = rows[0]["grid"]
    print(f"n={len(rows)} installed arms ({len(ctrl)} controls)")
    print(f"t_gen  median {med:.0f}  sd {y.std(ddof=1):.0f}  "
          f"grid {grid} ep  distinct values {len(set(y.tolist()))}")
    if ctrl:
        c = np.array([r["t"] for r in ctrl])
        print(f"no-install controls: median {np.median(c):.0f} "
              f"-> acceleration {np.median(c) / med:.1f}x")

    tot = ((y - y.mean()) ** 2).sum()
    for keys in (["s"], ["s", "base"]):
        e, cl = residualise(rows, y, keys)
        print(f"  R2 {'x'.join(keys):<10} ({len(cl):>2} cells) = "
              f"{1 - (e ** 2).sum() / tot:.3f}   resid sd {e.std(ddof=1):.0f} ep")

    e, cells = residualise(rows, y, ["s", "base"])
    sd = e.std(ddof=1)
    out = {"n": len(rows), "median_t_gen": med, "grid": grid,
           "resid_sd": float(sd), "band": args.band, "tests": {}}

    def run(name, stat, kind):
        obs, p, null = perm_test(e, cells, stat, rng)
        v = tost(obs, null, med, args.band)
        v.update(p=p, kind=kind)
        out["tests"][name] = v
        flag = ("EQUIVALENT" if v["equivalent"] else
                "EFFECT" if v["detected"] else "INCONCLUSIVE")
        print(f"  {name:<30} obs {obs:>8.3f}  ceil {v['noise_ceiling']:>8.3f}"
              f"  p {p:>5.3f}   {flag}")

    print("\nT1 content identity (crossed factor)")
    if rows[0]["content"] is not None and len({r["content"] for r in rows}) > 1:
        lab = np.array([r["content"] for r in rows])
        per = collections.Counter(lab.tolist())
        if min(per.values()) > 1:
            run("content identity", cat_stat(lab), "epochs")
        else:
            print("  SKIPPED: n=1 per content level — the factor is nested, "
                  "not crossed. This is exactly the Phase B defect.")
            out["tests"]["content identity"] = {"skipped": "nested, n=1/level"}
    else:
        print("  SKIPPED: manifest carries no shared content label.")
        out["tests"]["content identity"] = {"skipped": "no content label"}

    print("\nT2 content coordinates")
    for name, key in (("LP margin pctile", "lp_pct"),
                      ("LP relative margin", "lp_relM"),
                      ("mean target frequency", "meanf"),
                      ("target span", "spanf"),
                      ("lowest target frequency", "minf"),
                      ("max T_k rank of targets", "rankmax")):
        v = [r[key] for r in rows]
        if any(x is None for x in v) or len(set(v)) < 2:
            continue
        run(name, num_stat(v), "correlation")

    print("\nT3 Mantel (content similarity vs t_gen similarity)")
    pairs = [(i, j) for idx in cells.values()
             for i, j in itertools.combinations(idx, 2)]
    cd = np.array([1 - jaccard(set(rows[i]["S"]), set(rows[j]["S"]))
                   for i, j in pairs])
    inf = float((cd < 1).mean())
    if inf < 0.05:
        print(f"  SKIPPED: only {inf:.0%} of within-cell pairs share any "
              "frequency — no similarity gradient to test.")
        out["tests"]["mantel"] = {"skipped": "no similarity gradient",
                                  "informative_pairs": inf}
    else:
        ij = np.array(pairs)
        cdc = cd - cd.mean()
        cvar, crng = float((cdc ** 2).sum()), float(np.ptp(cd))

        def mantel(v):
            d = np.abs(v[ij[:, 0]] - v[ij[:, 1]])
            return abs(float(cdc @ d) / cvar) * crng

        run("mantel content-similarity", mantel, "correlation")
        out["tests"]["mantel content-similarity"]["informative_pairs"] = inf

    print("\nT4 realisation of the install")
    ex = [r["exact"] for r in rows]
    if any(x is not None for x in ex) and len(set(ex)) > 1:
        print(f"  install exact in {sum(bool(x) for x in ex)}/{len(ex)} arms; "
              f"median J(realised, installed) = "
              f"{np.median([r['J'] for r in rows]):.2f}")
        run("install exactness", cat_stat(np.array([float(bool(x))
                                                    for x in ex])), "epochs")
        run("J(realised, installed)", num_stat([r["J"] for r in rows]),
            "correlation")

    print("\npower")
    q = grid / np.sqrt(12)
    print(f"  residual sd {sd:.0f} ep on median t_gen {med:.0f} ep")
    print(f"  measurement grid {grid} ep -> quantisation sd {q:.0f} ep "
          f"({q / sd * 100:.0f}% of residual)")
    ncell = len(rows) / max(len({r['content'] for r in rows}), 2)
    mde = 2.8 * sd * np.sqrt(2 / ncell)
    print(f"  MDE (80% power, a={ALPHA}): {mde:.0f} ep = {mde / med * 100:.0f}% "
          f"of median t_gen   [band = {args.band * 100:.0f}%]")
    out["power"] = {"resid_sd": float(sd), "quantisation_sd": float(q),
                    "mde_epochs": float(mde), "mde_frac": float(mde / med)}
    if mde > args.band * med:
        print("  WARNING: MDE exceeds the equivalence band — this cohort "
              "cannot support an absence claim at the declared band.")
        out["power"]["underpowered"] = True

    verdicts = [v for v in out["tests"].values() if "equivalent" in v]
    out["verdict"] = ("EQUIVALENT" if verdicts and all(v["equivalent"] for v in verdicts)
                      else "EFFECT" if any(v["detected"] for v in verdicts)
                      else "INCONCLUSIVE")
    print(f"\nS-SIDE VERDICT: {out['verdict']} "
          f"({sum(v['equivalent'] for v in verdicts)}/{len(verdicts)} "
          "tests equivalent)")
    o = Path(args.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(out, indent=1, default=str))
    print(f"wrote {o}")


if __name__ == "__main__":
    main()
