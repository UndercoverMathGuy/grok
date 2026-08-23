"""EXPLORATORY drift dose-response (L-side v3): does learning ever rewrite
selection content during consolidation?

Per run (degrok cohort, spectra every 5 ep): find t_lock = first snapshot
where the detector committee is IDENTICAL on 3 consecutive snapshots (a
local-in-time anchor — deliberately NOT the final-committee-anchored
t_ident, which would bias drift toward zero by construction). Then:

  window  = t_gen - t_lock              (interventionally varied 36x by frac)
  drift   = 1 - J(committee@t_lock, committee@t_gen)
  events  = #membership changes (adds+drops) between t_lock and t_gen
  churn   = pre-lock control: median consecutive-snapshot J before t_lock
            (the instrument must see change where change happens)

Content-writing at rate r > 0 predicts drift rising with window; inertness
predicts drift flat at the instrument floor across the whole range.

Declared EXPLORATORY (data pre-existing); the confirmatory version is
preregistered as P-H8 in degrok2. Legacy AUC-based t_ident is reported
alongside for comparability.

Usage: python3 degrok/analyze_drift.py [--runs runs_degrok]
           [--out notes/drift_summary.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dial"))
from analyze_dial import committee_from_coeffs, jaccard, mw_auc  # noqa: E402


def sustained(epochs, ok):
    hits = np.where(ok[:-1] & ok[1:])[0]
    return int(hits[0]) if len(hits) else -1          # snapshot INDEX


def load_run(rd):
    cfg = json.loads((rd / "config.json").read_text())
    z = np.load(rd / "spectra.npz")
    ep = np.asarray(z["epochs"])
    C = np.asarray(z["coeffs"])
    comms = [committee_from_coeffs(C[s]) for s in range(len(ep))]

    s_gen = sustained(ep, np.asarray(z["test_acc"]) >= 0.99)
    # t_lock: identical committee on 3 consecutive snapshots (local anchor)
    s_lock = -1
    for s in range(len(comms) - 2):
        if comms[s] and comms[s] == comms[s + 1] == comms[s + 2]:
            s_lock = s
            break
    # legacy t_ident (final-committee-anchored AUC) for comparability
    nf = cfg["p"] // 2
    m = np.zeros(nf, bool)
    m[[k - 1 for k in comms[-1]]] = True
    E = np.asarray(z["energy"])
    auc = np.array([mw_auc(E[s], m) for s in range(len(ep))])
    s_ident = sustained(ep, auc >= 0.9)

    out = {"run": rd.parent.name + "/" + rd.name, "p": cfg["p"],
           "frac": cfg["frac_train"], "wd": cfg["weight_decay"],
           "t_lock": int(ep[s_lock]) if s_lock >= 0 else -1,
           "t_ident": int(ep[s_ident]) if s_ident >= 0 else -1,
           "t_gen": int(ep[s_gen]) if s_gen >= 0 else -1}
    if s_lock >= 0 and s_gen > s_lock:
        cl, cg = comms[s_lock], comms[s_gen]
        out["window"] = int(ep[s_gen] - ep[s_lock])
        out["drift"] = 1.0 - jaccard(cl, cg)
        out["adds"] = len(set(cg) - set(cl))
        out["drops"] = len(set(cl) - set(cg))
        # events across the whole window, not just endpoints
        ch = sum(comms[s] != comms[s + 1] for s in range(s_lock, s_gen))
        out["interior_changes"] = int(ch)
        # pre-lock churn control (consecutive-snapshot J before lock)
        pre = [jaccard(comms[s], comms[s + 1]) for s in range(max(1, s_lock))]
        out["prelock_median_J"] = float(np.median(pre)) if pre else None
        out["postlock_median_J"] = float(np.median(
            [jaccard(comms[s], comms[s + 1]) for s in range(s_lock, s_gen)]))
        # LATE drift, two instruments, measured from 20% into the window so
        # anchor-settling is excluded: the detector's discrete committee
        # (boundary flicker included) vs the fixed-K top energy set (K = the
        # final committee size; immune to gap-boundary flicker). The fixed-K
        # form is the content claim; the detector form bounds the flicker.
        w = ep[s_gen] - ep[s_lock]
        s20 = min(s_lock + int(np.searchsorted(ep[s_lock:s_gen],
                                               ep[s_lock] + 0.2 * w)), s_gen)
        Kf = len(cg)
        topK = lambda s: sorted((np.argsort(E[s])[::-1][:Kf] + 1).tolist())
        out["late_drift_det"] = 1.0 - jaccard(comms[s20], cg)
        out["late_drift_fixK"] = 1.0 - jaccard(topK(s20), topK(s_gen))
        out["edit_pos"] = [float((ep[s] - ep[s_lock]) / w)
                           for s in range(s_lock, s_gen)
                           if comms[s] != comms[s + 1]]
    return out


def med(v):
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs_degrok"))
    ap.add_argument("--out", default=str(ROOT / "notes" / "drift_summary.json"))
    args = ap.parse_args()

    rows = [load_run(sp.parent) for sp in
            sorted(Path(args.runs).glob("*/*/spectra.npz"))]
    ok = [r for r in rows if "drift" in r]
    print(f"{len(rows)} runs loaded, {len(ok)} with lock<gen windows\n")

    cells = {}
    for r in ok:
        cells.setdefault((r["p"], r["frac"], r["wd"]), []).append(r)
    print(f"{'p':>4} {'frac':>5} {'wd':>3} {'n':>3} {'window':>7} "
          f"{'drift':>6} {'any_chg':>7} {'endJ<1':>6} {'preJ':>5} {'postJ':>6}")
    for key, rs in sorted(cells.items()):
        p, f, wd = key
        print(f"{p:>4} {f:>5g} {wd:>3g} {len(rs):>3} "
              f"{med([r['window'] for r in rs]):>7.0f} "
              f"{med([r['drift'] for r in rs]):>6.3f} "
              f"{sum(r['interior_changes'] > 0 for r in rs):>7} "
              f"{sum(r['drift'] > 0 for r in rs):>6} "
              f"{med([r['prelock_median_J'] for r in rs]) or float('nan'):>5.2f} "
              f"{med([r['postlock_median_J'] for r in rs]):>6.3f}")

    w = np.array([r["window"] for r in ok], float)
    d = np.array([r["drift"] for r in ok], float)
    e = np.array([r["adds"] + r["drops"] for r in ok], float)
    lf = np.array([r["late_drift_fixK"] for r in ok], float)
    pos = np.concatenate([r["edit_pos"] for r in ok if r["edit_pos"]])
    stats = {
        "n": len(ok),
        "window_range": [float(w.min()), float(w.max())],
        "spearman_drift_vs_window": spearman(w, d),
        "spearman_events_vs_window": spearman(w, e),
        "runs_zero_endpoint_drift": int((d == 0).sum()),
        "runs_zero_interior_changes":
            int(sum(r["interior_changes"] == 0 for r in ok)),
        "median_prelock_J": med([r["prelock_median_J"] for r in ok]),
        "median_postlock_J": med([r["postlock_median_J"] for r in ok]),
        "late_drift_det_median": med([r["late_drift_det"] for r in ok]),
        "late_drift_fixK_median": float(np.median(lf)),
        "late_drift_fixK_zero_runs": int((lf == 0).sum()),
        "edit_pos_deciles":
            np.histogram(pos, bins=10, range=(0, 1))[0].tolist(),
        "late_fixK_by_cell": {
            f"p{p}_f{f:g}_wd{wd:g}": med([r["late_drift_fixK"] for r in rs])
            for (p, f, wd), rs in sorted(cells.items())},
    }
    print("\n" + json.dumps(stats, indent=1))

    o = Path(args.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps({"runs": rows, "stats": stats,
                             "status": "EXPLORATORY (pre-existing data); "
                             "confirmatory = degrok2 P-H8"},
                            indent=1, default=str))
    print(f"wrote {o}")


if __name__ == "__main__":
    main()
