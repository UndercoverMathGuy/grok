"""Analyzer for the clock-separation cohort (prereg: clock/README.md).

Paired per-seed deltas of t_ident and t_grok for each freeze arm against
the in-cohort base arm, committee Jaccards, manipulation checks, and the
P-T1..P-T4 verdicts. numpy+safetensors only; runs locally on pulled data.

Usage: python3 clock/analyze_clock.py [--runs runs_clock] [--out notes/clock_summary.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dial"))
from analyze_dial import committee_from_coeffs, grok_epoch, mw_auc, NF  # noqa: E402

NS = [50, 200, 500]
ARMS = ["base"] + [f"fz{n}" for n in NS] + [f"on{n}" for n in NS]


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def load_run(rd):
    z = np.load(rd / "spectra.npz")
    epochs = np.asarray(z["epochs"])
    E = np.asarray(z["energy"])
    comm = committee_from_coeffs(z["coeffs"][-1])
    mask = np.zeros(NF, bool)
    mask[[k - 1 for k in comm]] = True
    auc_t = np.array([mw_auc(E[s], mask) for s in range(len(epochs))])
    hits = np.where((auc_t[:-1] >= 0.9) & (auc_t[1:] >= 0.9))[0]
    return {
        "committee": list(comm), "epochs": epochs, "auc_t": auc_t,
        "train_acc": np.asarray(z["train_acc"]),
        "t_grok": int(grok_epoch(z)),
        "t_ident": int(epochs[hits[0]]) if len(hits) else -1,
    }


def med(v):
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs_clock"))
    ap.add_argument("--out", default=str(ROOT / "notes" / "clock_summary.json"))
    args = ap.parse_args()
    runs_root = Path(args.runs)

    data = {}  # (arm, seedkey) -> run record
    for arm in ARMS:
        for sp in sorted((runs_root / arm).glob("*/spectra.npz")):
            data[(arm, sp.parent.name)] = load_run(sp.parent)
    seeds = sorted({k for a, k in data if a == "base"})
    print(f"{len(data)} runs loaded, {len(seeds)} seed pairs")

    out = {"arms": {}, "verdicts": {}}
    rows = []
    for arm in ARMS:
        recs = [(k, data.get((arm, k))) for k in seeds]
        n_grok = sum(1 for _, r in recs if r and r["t_grok"] >= 0)
        d_ti, d_tg, js = [], [], []
        for k, r in recs:
            b = data.get(("base", k))
            if not r or not b:
                continue
            js.append(jaccard(r["committee"], b["committee"]))
            if r["t_ident"] >= 0 and b["t_ident"] >= 0:
                d_ti.append(r["t_ident"] - b["t_ident"])
            if r["t_grok"] >= 0 and b["t_grok"] >= 0:
                d_tg.append(r["t_grok"] - b["t_grok"])
        row = {"arm": arm, "n_grok": n_grok,
               "t_ident": med([r["t_ident"] for _, r in recs
                               if r and r["t_ident"] >= 0]),
               "t_grok": med([r["t_grok"] for _, r in recs
                              if r and r["t_grok"] >= 0]),
               "d_t_ident": med(d_ti), "d_t_grok": med(d_tg),
               "d_ti_all": d_ti, "d_tg_all": d_tg,
               "J_base": med(js)}
        rows.append(row)
        out["arms"][arm] = {k: v for k, v in row.items()}

    print(f"\n{'arm':>6} {'n_grok':>6} {'t_ident':>8} {'t_grok':>7} "
          f"{'d_ident':>8} {'d_grok':>7} {'J_base':>7}")
    for r in rows:
        f = lambda k: format(r[k], ".0f") if r[k] is not None else "-"
        print(f"{r['arm']:>6} {r['n_grok']:>6} {f('t_ident'):>8} "
              f"{f('t_grok'):>7} {f('d_t_ident'):>8} {f('d_t_grok'):>7} "
              f"{r['J_base']:>7.2f}" if r['J_base'] is not None else "")

    byarm = {r["arm"]: r for r in rows}

    # P-T1: dose-response of the selection delay
    pts = [(n, d) for n in NS for d in byarm[f"fz{n}"]["d_ti_all"]]
    verdict = {}
    if pts:
        x = np.array([p[0] for p in pts], float)
        y = np.array([p[1] for p in pts], float)
        rx, ry = x.argsort().argsort(), y.argsort().argsort()
        rho = float(np.corrcoef(rx, ry)[0, 1])
        rng = np.random.default_rng(0)
        p = (sum(abs(np.corrcoef(rng.permutation(rx), ry)[0, 1]) >= abs(rho)
                 for _ in range(20000)) + 1) / 20001
        ok_med = all((byarm[f"fz{n}"]["d_t_ident"] or -1e9) >= 0.5 * n
                     for n in (200, 500))
        kill = (byarm["fz500"]["d_t_ident"] or 0) < 125
        verdict = {"rho": rho, "p": p, "medians_ok": ok_med, "kill": kill,
                   "result": ("KILL — selection not W_E-clocked" if kill else
                              "PASS" if ok_med and rho > 0 and p < 0.05 else
                              "FAIL/partial")}
    out["verdicts"]["P-T1"] = verdict
    print(f"\nP-T1 (fz delay dose-response): {verdict}")

    # P-T2: identity survives the pause (+ shuffled-pair null)
    j5 = [jaccard(data[("fz500", k)]["committee"], data[("base", k)]["committee"])
          for k in seeds if ("fz500", k) in data]
    null = [jaccard(data[("fz500", a)]["committee"], data[("base", b)]["committee"])
            for a in seeds for b in seeds
            if a != b and ("fz500", a) in data]
    v2 = {"J_median": med(j5), "null_median": med(null),
          "result": "PASS" if (med(j5) or 0) >= 0.6 and med(j5) > med(null)
          else "weakened — delay scrambles identity"}
    out["verdicts"]["P-T2"] = v2
    print(f"P-T2 (identity survives): {v2}")

    # P-T3: clock relationship at N=500 (branch, not gated)
    dti, dtg = byarm["fz500"]["d_t_ident"], byarm["fz500"]["d_t_grok"]
    if dti:
        ratio = (dtg or 0) / dti if dti else None
        branch = ("independent clocks" if (dtg or 0) <= 0.5 * dti else
                  "serial gating — grok WAITS for selection"
                  if 0.75 <= ratio <= 1.25 else f"partial coupling ({ratio:.2f})")
        out["verdicts"]["P-T3"] = {"d_t_ident": dti, "d_t_grok": dtg,
                                   "branch": branch}
        print(f"P-T3 (clock relationship): d_ident {dti:.0f}  d_grok {dtg}  -> {branch}")

    # P-T4: mirror — on500 delays grok, not selection
    dti4, dtg4 = byarm["on500"]["d_t_ident"], byarm["on500"]["d_t_grok"]
    v4 = {"d_t_ident": dti4, "d_t_grok": dtg4,
          "result": "PASS" if (dtg4 or 0) >= 250 and (dti4 is not None and dti4 <= 100)
          else "FAIL/partial (see prereg caveat on dim readout)"}
    out["verdicts"]["P-T4"] = v4
    print(f"P-T4 (mirror): {v4}")

    # manipulation checks
    for n in NS:
        diffs = []
        for k in seeds:
            r, b = data.get((f"fz{n}", k)), data.get(("base", k))
            if r is None or b is None:
                continue
            w = r["epochs"] < n
            diffs.append(float(np.abs(r["train_acc"][w] - b["train_acc"][w]).max()))
        on = data.get((f"on{n}", seeds[0]))
        on_acc = (float(on["train_acc"][on["epochs"] < n].max())
                  if on is not None and (on["epochs"] < n).any() else None)
        print(f"check fz{n}: max |train_acc - base| during window: "
              f"median {med(diffs):.3f}   on{n} peak train_acc in window: {on_acc}")
        out.setdefault("checks", {})[f"N{n}"] = {"fz_acc_dev": med(diffs),
                                                 "on_acc_peak": on_acc}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {k: v for k, v in out.items()}, indent=1, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
