"""Early identifiability of the final committee in the logit spectrum,
per width — the free (analysis-only) warm-up test of lazy selection.

Mini-prereg (written BEFORE running, 2026-08-14):
- SUPPORT: at matched training phase (snapshot nearest t = 0.2*t_grok,
  strictly pre-grok), the logit-spectrum AUC for the run's FINAL committee
  rises with width (Spearman over per-run points at wd=1.0, p<0.05).
  The same direction at fixed epoch 500 corroborates, but fixed-epoch
  comparisons are confounded by grok-time shifts across width, so the
  phase-matched number is the scored one.
- KILL: flat or negative width trend at matched phase -> the cheap support
  for lazy selection fails and the linearized-twin experiment loses its
  motivation; the mechanism claim reverts to open.

Known baseline going in: the committee is readable from WEIGHTS at epoch 0
(T_k) but logits are blind at init — identifiability must emerge during
memorization. The question is whether it emerges earlier/stronger with
width.

Usage: python3 dial/early_ident.py [--runs runs_dial] [--out notes/dial_early_ident.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from analyze_dial import committee_from_coeffs, grok_epoch, mw_auc, spearman_perm, NF  # noqa: E402

FIXED_EPOCHS = [100, 300, 500, 1000, 3000]
PHASES = [0.1, 0.2, 0.5]


def load_run(rd):
    cfg = json.loads((rd / "config.json").read_text())
    z = np.load(rd / "spectra.npz")
    tg = grok_epoch(z)
    if tg < 0:
        return None
    comm = committee_from_coeffs(z["coeffs"][-1])
    mask = np.zeros(NF, bool)
    mask[[k - 1 for k in comm]] = True
    epochs = np.asarray(z["epochs"])
    E = np.asarray(z["energy"])
    auc_t = np.array([mw_auc(E[s], mask) for s in range(len(epochs))])
    row = {"run": str(rd), "d_mlp": cfg["d_mlp"], "wd": cfg["weight_decay"],
           "t_grok": tg, "committee": comm}
    for ep in FIXED_EPOCHS:
        row[f"auc_ep{ep}"] = float(auc_t[np.argmin(np.abs(epochs - ep))])
    for ph in PHASES:
        s = np.argmin(np.abs(epochs - ph * tg))
        if epochs[s] >= tg:  # keep it strictly pre-grok
            s = max(0, np.searchsorted(epochs, tg) - 1)
        row[f"auc_phase{ph:g}"] = float(auc_t[s])
    # first epoch with AUC >= 0.9 sustained for the following snapshot too
    hits = np.where((auc_t[:-1] >= 0.9) & (auc_t[1:] >= 0.9))[0]
    row["t_ident"] = int(epochs[hits[0]]) if len(hits) else -1
    row["ident_frac"] = (row["t_ident"] / tg) if row["t_ident"] >= 0 else None
    return row


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--runs", default=str(root / "runs_dial"))
    ap.add_argument("--out", default=str(root / "notes" / "dial_early_ident.json"))
    args = ap.parse_args()

    rows = [r for r in (load_run(p.parent) for p in
                        Path(args.runs).glob("*/*/spectra.npz")) if r]
    print(f"{len(rows)} groked runs")

    cells = {}
    for r in rows:
        cells.setdefault((r["d_mlp"], r["wd"]), []).append(r)
    keys = (["t_grok", "t_ident", "ident_frac"]
            + [f"auc_ep{e}" for e in FIXED_EPOCHS]
            + [f"auc_phase{p:g}" for p in PHASES])
    table = []
    for (w, wd), rs in sorted(cells.items()):
        row = {"d_mlp": w, "wd": wd, "n": len(rs)}
        for k in keys:
            vals = [r[k] for r in rs if r[k] is not None and r[k] != -1]
            row[k] = float(np.mean(vals)) if vals else None
        table.append(row)

    print(f"\n{'d_mlp':>6} {'wd':>5} {'n':>3} {'t_grok':>7} {'t_ident':>8} "
          f"{'i_frac':>7} {'auc@500':>8} {'auc@.1g':>8} {'auc@.2g':>8} {'auc@.5g':>8}")
    for c in table:
        f = lambda k, s: format(c[k], s) if c[k] is not None else "-"
        print(f"{c['d_mlp']:>6} {c['wd']:>5g} {c['n']:>3} {f('t_grok','7.0f'):>7} "
              f"{f('t_ident','8.0f'):>8} {f('ident_frac','7.3f'):>7} "
              f"{f('auc_ep500','8.3f'):>8} {f('auc_phase0.1','8.3f'):>8} "
              f"{f('auc_phase0.2','8.3f'):>8} {f('auc_phase0.5','8.3f'):>8}")

    tests = {}
    for wd in (0.3, 1.0, 3.0):
        sel = [r for r in rows if r["wd"] == wd]
        for key in ("auc_phase0.2", "ident_frac"):
            pts = [(r["d_mlp"], r[key]) for r in sel if r[key] is not None]
            if len(pts) < 6:
                continue
            rho, p = spearman_perm([x for x, _ in pts], [y for _, y in pts])
            tests[f"{key}_vs_width_wd{wd:g}"] = {
                "n": len(pts), "spearman": rho, "p_perm": p}
    print("\nwidth trends (per-run points):")
    for name, t in tests.items():
        print(f"  {name}: n {t['n']}  rho {t['spearman']:+.3f}  p {t['p_perm']:.4f}")
    scored = tests.get("auc_phase0.2_vs_width_wd1")
    if scored:
        v = ("SUPPORT" if scored["spearman"] > 0 and scored["p_perm"] < 0.05
             else "KILL/flat")
        print(f"\nmini-prereg verdict (auc@0.2*t_grok vs width, wd=1.0): {v}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"runs": rows, "cells": table, "tests": tests},
                              indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()