"""Race analysis: fine-grained (every-5-epoch) early spectra from runs_race
joined against the STORED 50k-epoch final committees from runs_dial.

This is the powered version of early_ident.py: instead of only the 32 runs
that grok inside the 2000-epoch race (wd=3 only), every race run whose
stored 50k twin groked gets a label — the trajectories are the same seeds on
the same trainer (verified 27/32 exact committee match at matched epoch
2000; remaining 5 differ by one borderline gap-detector member).

Status: exploratory follow-up, NOT pre-registered — the mini-prereg in
early_ident.py was scored (KILL/flat, instrument saturated) before this
was written, and the 32-run race summary was seen first. Every p-value
here is descriptive.

Questions:
- Q1 (epoch-0): does AUC of the untrained network's logit spectrum vs the
  final committee rise with width?  Unit = init (width, ds, is) — epoch-0
  logits are identical across wd, so wd copies are averaged, not pooled.
- Q2 (lock-in time): does t_ident (first sustained AUC>=0.9) fall with
  width?  Tested within-wd (width labels permuted inside each wd stratum)
  so wd's huge effect on speed cannot masquerade as a width effect.
- Q3 (early AUC): same stratified test on AUC at fixed epochs 5..100.

Usage: python3 dial/race_analyze.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from analyze_dial import committee_from_coeffs, grok_epoch, mw_auc, NF  # noqa: E402

FIXED = [0, 5, 10, 20, 50, 100, 500, 2000]


def spearman_strat(x, y, strata, n_perm=20_000, seed=0):
    """Spearman rho with a permutation test that shuffles x only within
    each stratum (kills between-stratum confounds, e.g. wd -> speed)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    strata = np.asarray(strata)
    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    obs = np.corrcoef(rx, ry)[0, 1]
    rng = np.random.default_rng(seed)
    hits = 0
    xp = x.copy()
    for _ in range(n_perm):
        for s in np.unique(strata):
            m = strata == s
            xp[m] = rng.permutation(xp[m])
        r = np.corrcoef(xp.argsort().argsort().astype(float), ry)[0, 1]
        if abs(r) >= abs(obs):
            hits += 1
    return float(obs), (hits + 1) / (n_perm + 1)


def main():
    rows = []
    for sp in sorted((ROOT / "runs_race").glob("*/*/spectra.npz")):
        rd = sp.parent
        rel = rd.relative_to(ROOT / "runs_race")
        stored = ROOT / "runs_dial" / rel / "spectra.npz"
        zs = np.load(stored)
        tg = grok_epoch(zs)
        if tg < 0:  # stored twin never groked (some wd=0.1) -> no label
            continue
        comm = committee_from_coeffs(zs["coeffs"][-1])
        mask = np.zeros(NF, bool)
        mask[[k - 1 for k in comm]] = True

        cfg = json.loads((rd / "config.json").read_text())
        z = np.load(sp)
        epochs = np.asarray(z["epochs"])
        E = np.asarray(z["energy"])
        auc_t = np.array([mw_auc(E[s], mask) for s in range(len(epochs))])

        row = {"run": str(rel), "d_mlp": cfg["d_mlp"], "wd": cfg["weight_decay"],
               "ds": cfg["data_seed"], "is": cfg["init_seed"],
               "t_grok_stored": int(tg), "committee": list(comm)}
        for ep in FIXED:
            row[f"auc_ep{ep}"] = float(auc_t[np.argmin(np.abs(epochs - ep))])
        hits = np.where((auc_t[:-1] >= 0.9) & (auc_t[1:] >= 0.9))[0]
        row["t_ident"] = int(epochs[hits[0]]) if len(hits) else -1
        row["ident_frac"] = (row["t_ident"] / tg) if row["t_ident"] >= 0 else None
        rows.append(row)
    print(f"{len(rows)} labeled runs (stored twin groked)")

    # per-cell table
    cells = {}
    for r in rows:
        cells.setdefault((r["d_mlp"], r["wd"]), []).append(r)
    hdr = (f"{'d_mlp':>6} {'wd':>5} {'n':>3} {'t_grok':>7} {'t_ident':>8} "
           f"{'auc@0':>7} {'auc@5':>7} {'auc@10':>7} {'auc@20':>7} {'auc@50':>7} {'auc@100':>8}")
    print("\n" + hdr)
    for (w, wd), rs in sorted(cells.items()):
        ti = [r["t_ident"] for r in rs if r["t_ident"] >= 0]
        f = lambda k: np.mean([r[k] for r in rs])
        print(f"{w:>6} {wd:>5g} {len(rs):>3} {np.mean([r['t_grok_stored'] for r in rs]):>7.0f} "
              f"{(np.mean(ti) if ti else float('nan')):>8.1f} "
              f"{f('auc_ep0'):>7.3f} {f('auc_ep5'):>7.3f} {f('auc_ep10'):>7.3f} "
              f"{f('auc_ep20'):>7.3f} {f('auc_ep50'):>7.3f} {f('auc_ep100'):>8.3f}")

    tests = {}

    # Q1: epoch-0 AUC vs width, unit = init (avg over wd labelings)
    inits = {}
    for r in rows:
        inits.setdefault((r["d_mlp"], r["ds"], r["is"]), []).append(r["auc_ep0"])
    iw = [w for (w, _, _) in inits]
    ia = [float(np.mean(v)) for v in inits.values()]
    rho, p = spearman_strat(iw, ia, np.zeros(len(iw)))
    tests["q1_auc_ep0_vs_width_initlevel"] = {"n": len(iw), "rho": rho, "p": p}
    print(f"\nQ1 epoch-0 AUC vs width (init-level, n={len(iw)}): "
          f"rho {rho:+.3f}  p {p:.4f}")
    for w in sorted(set(iw)):
        sel = [a for ww, a in zip(iw, ia) if ww == w]
        print(f"   width {w:>4}: mean auc@0 {np.mean(sel):.3f}  "
              f"frac>=0.9 {np.mean([a >= 0.9 for a in sel]):.2f}  n {len(sel)}")

    # Q2: t_ident vs width, within-wd permutation
    sel = [r for r in rows if r["t_ident"] >= 0]
    rho, p = spearman_strat([r["d_mlp"] for r in sel], [r["t_ident"] for r in sel],
                            [r["wd"] for r in sel])
    tests["q2_t_ident_vs_width_stratwd"] = {"n": len(sel), "rho": rho, "p": p}
    print(f"\nQ2 t_ident vs width (within-wd perm, n={len(sel)}): "
          f"rho {rho:+.3f}  p {p:.4f}")
    sel2 = [r for r in sel if r["ident_frac"] is not None]
    rho, p = spearman_strat([r["d_mlp"] for r in sel2], [r["ident_frac"] for r in sel2],
                            [r["wd"] for r in sel2])
    tests["q2b_ident_frac_vs_width_stratwd"] = {"n": len(sel2), "rho": rho, "p": p}
    print(f"    ident_frac vs width (within-wd perm, n={len(sel2)}): "
          f"rho {rho:+.3f}  p {p:.4f}")

    # Q3: fixed-epoch AUC vs width, within-wd permutation
    print()
    for ep in [5, 10, 20, 50, 100]:
        rho, p = spearman_strat([r["d_mlp"] for r in rows],
                                [r[f"auc_ep{ep}"] for r in rows],
                                [r["wd"] for r in rows])
        tests[f"q3_auc_ep{ep}_vs_width_stratwd"] = {"n": len(rows), "rho": rho, "p": p}
        print(f"Q3 auc@{ep:<3} vs width (within-wd perm, n={len(rows)}): "
              f"rho {rho:+.3f}  p {p:.4f}")

    out = ROOT / "notes" / "race_summary.json"
    out.write_text(json.dumps({"runs": rows, "tests": tests}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
