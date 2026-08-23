"""Analyzer for the de-grok cohort (prereg: degrok/README.md).

Per run: t_ident (own final committee, AUC>=0.9 sustained 2 snapshots),
t_gen (test acc >= 0.99 sustained 2 snapshots), epoch-0 W_E tilt AUC.
Cells: (p, frac, wd). Scored: P-G1..P-G5 + the pre-declared branch.
p-parameterized (p=127 anchor batch has 63 frequencies, not 56).

Usage: python3 degrok/analyze_degrok.py [--runs runs_degrok] [--out notes/degrok_summary.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dial"))
from analyze_dial import committee_from_coeffs, fourier_basis, mw_auc, jaccard  # noqa: E402

FRACS = [0.3, 0.5, 0.7, 0.9]
_BASIS = {}


def tilt_auc0(rd, p, comm):
    if p not in _BASIS:
        _BASIS[p] = fourier_basis(p)
    ck = load_file(str(rd / "checkpoints" / "epoch_00000.safetensors"))
    W_E = ck["embed.W_E"].astype(np.float64)[:, :p]
    E = ((_BASIS[p] @ W_E.T) ** 2).sum(1)
    nf = p // 2
    t = E[1::2][:nf] + E[2::2][:nf]
    m = np.zeros(nf, bool)
    m[[k - 1 for k in comm]] = True
    return mw_auc(t, m)


def sustained(epochs, ok):
    hits = np.where(ok[:-1] & ok[1:])[0]
    return int(epochs[hits[0]]) if len(hits) else -1


def load_run(rd):
    cfg = json.loads((rd / "config.json").read_text())
    z = np.load(rd / "spectra.npz")
    epochs = np.asarray(z["epochs"])
    t_gen = sustained(epochs, np.asarray(z["test_acc"]) >= 0.99)
    comm = committee_from_coeffs(z["coeffs"][-1])
    nf = cfg["p"] // 2
    m = np.zeros(nf, bool)
    m[[k - 1 for k in comm]] = True
    E = np.asarray(z["energy"])
    auc_t = np.array([mw_auc(E[s], m) for s in range(len(epochs))])
    return {"run": rd.name, "p": cfg["p"], "frac": cfg["frac_train"],
            "wd": cfg["weight_decay"], "ds": cfg["data_seed"],
            "is": cfg["init_seed"], "K": len(comm), "committee": comm,
            "t_gen": t_gen,
            "t_ident": sustained(epochs, auc_t >= 0.9),
            "tilt_auc0": tilt_auc0(rd, cfg["p"], comm)}


def med(v):
    v = [x for x in v if x is not None and x >= 0]
    return float(np.median(v)) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs_degrok"))
    ap.add_argument("--out", default=str(ROOT / "notes" / "degrok_summary.json"))
    args = ap.parse_args()

    rows = [load_run(sp.parent) for sp in
            sorted(Path(args.runs).glob("*/*/spectra.npz"))]
    print(f"{len(rows)} runs loaded")

    cells = {}
    for r in rows:
        cells.setdefault((r["p"], r["frac"], r["wd"]), []).append(r)
    print(f"\n{'p':>4} {'frac':>5} {'wd':>4} {'n':>3} {'n_gen':>5} "
          f"{'t_ident':>8} {'t_gen':>7} {'tiltAUC0':>9} {'K':>4}")
    for (p, f, wd), rs in sorted(cells.items()):
        ti, tg = med([r["t_ident"] for r in rs]), med([r["t_gen"] for r in rs])
        print(f"{p:>4} {f:>5g} {wd:>4g} {len(rs):>3} "
              f"{sum(r['t_gen'] >= 0 for r in rs):>5} "
              f"{ti if ti is not None else float('nan'):>8.0f} "
              f"{tg if tg is not None else float('nan'):>7.0f} "
              f"{np.mean([r['tilt_auc0'] for r in rs]):>9.3f} "
              f"{np.mean([r['K'] for r in rs]):>4.1f}")

    m = lambda p, f, wd, k: med([r[k] for r in cells.get((p, f, wd), [])])
    V = {}

    ti03, ti09 = m(113, 0.3, 1.0, "t_ident"), m(113, 0.9, 1.0, "t_ident")
    V["P-G1"] = {"t_ident_f0.3": ti03, "t_ident_f0.9": ti09,
                 "ratio": ti09 / ti03 if ti03 else None,
                 "result": ("KILL" if ti03 and ti09 / ti03 >= 5 else
                            "PASS" if ti03 and ti09 / ti03 <= 3 else "GRAY")}

    g2 = {}
    for f in FRACS:
        a, b = m(113, f, 1.0, "t_ident"), m(113, f, 3.0, "t_ident")
        g2[f] = None if a is None or b is None else abs(a - b)
    tg1, tg3 = m(113, 0.3, 1.0, "t_gen"), m(113, 0.3, 3.0, "t_gen")
    V["P-G2"] = {"abs_ti_diff_by_frac": g2,
                 "tgen_ratio_f0.3": tg1 / tg3 if tg3 else None,
                 "result": "PASS" if all(d is not None and d <= 25
                                         for d in g2.values())
                 and tg3 and tg1 / tg3 >= 2 else "FAIL/partial"}

    tis = [m(113, f, 1.0, "t_ident") for f in FRACS]
    tgs = [m(113, f, 1.0, "t_gen") for f in FRACS]
    if all(x is not None for x in tis + tgs):
        V["P-G3"] = {"t_ident_by_frac": tis, "t_gen_by_frac": tgs,
                     "tgen_spread": max(tgs) / min(tgs),
                     "tident_spread": max(tis) / min(tis),
                     "result": "PASS" if max(tgs) / min(tgs) >= 5
                     and max(tis) / min(tis) < 3 else "FAIL/partial"}

    a4 = {f"{f}/{wd}": float(np.mean([r["tilt_auc0"] for r in
                                      cells.get((113, f, wd), [])] or [np.nan]))
          for f in FRACS for wd in (1.0, 3.0)}
    V["P-G4"] = {"tilt_auc0": a4,
                 "result": "PASS" if all(v >= 0.65 for v in a4.values())
                 else "FAIL/partial"}

    a, b = m(127, 0.3, 1.0, "t_ident"), m(127, 0.3, 3.0, "t_ident")
    p127auc = float(np.mean([r["tilt_auc0"] for r in rows if r["p"] == 127]
                            or [np.nan]))
    V["P-G5"] = {"ti_wd1": a, "ti_wd3": b, "tilt_auc0": p127auc,
                 "result": "PASS" if a is not None and b is not None
                 and abs(a - b) <= 25 and p127auc >= 0.65 else "FAIL/partial"}

    if ti09 is not None:
        tg09 = m(113, 0.9, 1.0, "t_gen")
        V["branch_clocks_cross"] = (tg09 is not None and tg09 <= ti09)

    print()
    for k, v in V.items():
        print(f"{k}: {v}")

    # exploratory: cross-frac committee overlap for matched (ds, is), wd=1
    print("\nexploratory cross-frac committee J (vs frac 0.3, wd 1, p 113):")
    base = {(r["ds"], r["is"]): r["committee"]
            for r in cells.get((113, 0.3, 1.0), [])}
    for f in FRACS[1:]:
        js = [jaccard(r["committee"], base[(r["ds"], r["is"])])
              for r in cells.get((113, f, 1.0), []) if (r["ds"], r["is"]) in base]
        print(f"  frac {f}: median J {np.median(js) if js else float('nan'):.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"runs": rows, "verdicts": V}, indent=1,
                              default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
