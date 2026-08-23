"""Analyzer for the multi-task cohort (prereg: tasks/README.md).

Usage: python3 tasks/analyze_tasks.py [--runs runs_tasks] [--out notes/tasks_summary.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dial"))
sys.path.insert(0, str(ROOT / "cloud"))
from analyze_dial import committee_from_coeffs, fourier_basis, mw_auc  # noqa: E402

_BASIS = {}


def tilt_energy(rd, cfg):
    ck = load_file(str(rd / "checkpoints" / "epoch_00000.safetensors"))
    W_E = ck["embed.W_E"].astype(np.float64)[:, :cfg["p"]]
    if cfg["fn_name"] == "s5":
        from groups import irrep_energy
        return irrep_energy(W_E).sum(0)[1:]          # (6,) nontrivial
    p = cfg["p"]
    if p not in _BASIS:
        _BASIS[p] = fourier_basis(p)
    E = ((_BASIS[p] @ W_E.T) ** 2).sum(1)
    return E[1::2][:p // 2] + E[2::2][:p // 2]


def sustained(ep, ok):
    h = np.where(ok[:-1] & ok[1:])[0]
    return int(ep[h[0]]) if len(h) else -1


def load_run(rd):
    cfg = json.loads((rd / "config.json").read_text())
    z = np.load(rd / "spectra.npz")
    ep = np.asarray(z["epochs"])
    comm = committee_from_coeffs(z["coeffs"][-1])
    nf = z["energy"].shape[1]
    m = np.zeros(nf, bool)
    m[[k - 1 for k in comm]] = True
    auc_t = np.array([mw_auc(np.asarray(z["energy"][s]), m)
                      for s in range(len(ep))])
    t = tilt_energy(rd, cfg)
    return {"run": rd.parent.name + "/" + rd.name, "task": cfg["fn_name"],
            "p": cfg["p"], "frac": cfg["frac_train"], "wd": cfg["weight_decay"],
            "K": len(comm), "nf": nf, "committee": comm,
            "t_gen": sustained(ep, np.asarray(z["test_acc"]) >= 0.99),
            "t_ident": sustained(ep, auc_t >= 0.9),
            "tilt_auc0": mw_auc(t, m)}


def med(v):
    v = [x for x in v if x is not None and x >= 0]
    return float(np.median(v)) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs_tasks"))
    ap.add_argument("--out", default=str(ROOT / "notes" / "tasks_summary.json"))
    args = ap.parse_args()

    rows = [load_run(sp.parent) for sp in
            sorted(Path(args.runs).glob("*/*/spectra.npz"))]
    cells = {}
    for r in rows:
        cells.setdefault(r["run"].split("/")[0], []).append(r)

    print(f"{len(rows)} runs\n")
    print(f"{'cell':>18} {'n':>3} {'grok':>5} {'t_ident':>8} {'t_gen':>7} "
          f"{'ti/tg':>6} {'K':>5} {'auc0':>6}")
    table = {}
    for cell, rs in sorted(cells.items()):
        ti, tg = med([r["t_ident"] for r in rs]), med([r["t_gen"] for r in rs])
        gr = np.mean([r["t_gen"] >= 0 for r in rs])
        row = {"n": len(rs), "grok_rate": float(gr), "t_ident": ti, "t_gen": tg,
               "K": float(np.mean([r["K"] for r in rs])),
               "nf": rs[0]["nf"],
               "tilt_auc0": float(np.mean([r["tilt_auc0"] for r in rs]))}
        table[cell] = row
        f = lambda v, s: format(v, s) if v is not None else "-"
        print(f"{cell:>18} {len(rs):>3} {gr:>5.2f} {f(ti, '8.0f'):>8} "
              f"{f(tg, '7.0f'):>7} "
              f"{f(ti / tg if ti and tg else None, '6.3f'):>6} "
              f"{row['K']:>5.1f} {row['tilt_auc0']:>6.3f}")

    V = {}
    for task, kmax in (("s5", 4), ("m96", 10)):
        tc = {c: t for c, t in table.items() if c.startswith(task)
              and t["grok_rate"] >= 0.5}
        if not tc:
            V[task] = {"verdict": "unmeasurable (no cell groks >= 50%)"}
            continue
        p1 = all(t["K"] <= kmax for t in tc.values())
        p2 = all(t["t_ident"] is not None and t["t_gen"]
                 and t["t_ident"] <= 0.2 * t["t_gen"] for t in tc.values())
        p4 = all(t["tilt_auc0"] >= 0.6 for t in tc.values())
        # P-M3: within-frac wd pairs
        p3, pairs = None, []
        byfrac = {}
        for c, t in tc.items():
            frac = c.split("_wd")[0]
            byfrac.setdefault(frac, []).append((float(c.split("_wd")[1]), t))
        for frac, wds in byfrac.items():
            if len(wds) < 2:
                continue
            wds.sort()
            lo, hi = wds[0][1], wds[-1][1]
            if lo["t_gen"] and hi["t_gen"] and lo["t_ident"] and hi["t_ident"]:
                rg = lo["t_gen"] / hi["t_gen"]
                ri = lo["t_ident"] / hi["t_ident"]
                pairs.append({"frac": frac, "tgen_ratio": rg, "ti_ratio": ri,
                              "ok": rg >= 1.5 and ri <= 1.25})
        if pairs:
            p3 = all(p["ok"] for p in pairs)
        V[task] = {"P-M1_sparse": p1, "P-M2_early": p2, "P-M3_wd": p3,
                   "P-M3_pairs": pairs, "P-M4_readable": p4,
                   "passes": bool(p1 and p2 and p4 and (p3 is not False))}
    print()
    for task, v in V.items():
        print(f"{task}: {v}")
    gen = sum(1 for v in V.values() if v.get("passes"))
    print(f"\nGENERALITY: {gen}/2 tasks pass "
          f"({'claim supported' if gen >= 1 else 'mod-add-scoped'})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"runs": rows, "cells": table, "verdicts": V},
                              indent=1, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
