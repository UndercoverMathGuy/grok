"""Analyzer for de-grok 2 (prereg: degrok2/README.md, P-H1..P-H7).

Cells: (fn, p, d_mlp, frac, wd). Same metrics as degrok v1; tilt_auc0
skipped for s5 (irrep instrument stays exploratory in tasks/).

Usage: python3 degrok2/analyze_degrok2.py [--runs runs_degrok2]
           [--out notes/degrok2_summary.json]
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

A_FRACS = [0.2, 0.25, 0.3, 0.4, 0.6, 0.8, 0.9]
B_FRACS = [0.3, 0.6, 0.9]
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
    comm = committee_from_coeffs(z["coeffs"][-1])
    nf = z["energy"].shape[1]
    m = np.zeros(nf, bool)
    m[[k - 1 for k in comm]] = True
    E = np.asarray(z["energy"])
    auc_t = np.array([mw_auc(E[s], m) for s in range(len(epochs))])
    fn = cfg["fn_name"]
    return {"run": rd.name, "fn": fn, "p": cfg["p"], "m": cfg["d_mlp"],
            "frac": cfg["frac_train"], "wd": cfg["weight_decay"],
            "ds": cfg["data_seed"], "is": cfg["init_seed"],
            "K": len(comm), "committee": comm,
            "t_gen": sustained(epochs, np.asarray(z["test_acc"]) >= 0.99),
            "t_ident": sustained(epochs, auc_t >= 0.9),
            "tilt_auc0": (None if fn == "s5"
                          else tilt_auc0(rd, cfg["p"], comm))}


def med(v):
    v = [x for x in v if x is not None and x >= 0]
    return float(np.median(v)) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs_degrok2"))
    ap.add_argument("--out", default=str(ROOT / "notes" / "degrok2_summary.json"))
    args = ap.parse_args()

    rows = [load_run(sp.parent) for sp in
            sorted(Path(args.runs).glob("*/*/spectra.npz"))]
    print(f"{len(rows)} runs loaded")

    cells = {}
    for r in rows:
        cells.setdefault((r["fn"], r["p"], r["m"], r["frac"], r["wd"]),
                         []).append(r)

    def gen_ok(rs):                      # cell qualifies: >= half generalize
        return sum(r["t_gen"] >= 0 for r in rs) * 2 >= len(rs)

    print(f"\n{'fn':>4} {'p':>4} {'m':>5} {'frac':>5} {'wd':>4} {'n':>3} "
          f"{'n_gen':>5} {'t_ident':>8} {'t_gen':>7} {'tilt0':>6} {'K':>4}")
    for key, rs in sorted(cells.items()):
        fn, p, m_, f, wd = key
        ti, tg = med([r["t_ident"] for r in rs]), med([r["t_gen"] for r in rs])
        tilts = [r["tilt_auc0"] for r in rs if r["tilt_auc0"] is not None]
        print(f"{fn:>4} {p:>4} {m_:>5} {f:>5g} {wd:>4g} {len(rs):>3} "
              f"{sum(r['t_gen'] >= 0 for r in rs):>5} "
              f"{ti if ti is not None else float('nan'):>8.0f} "
              f"{tg if tg is not None else float('nan'):>7.0f} "
              f"{np.mean(tilts) if tilts else float('nan'):>6.3f} "
              f"{np.mean([r['K'] for r in rs]):>4.1f}")

    C = lambda fn, p, m_, f, wd: cells.get((fn, p, m_, f, wd), [])
    M = lambda fn, p, m_, f, wd, k: med([r[k] for r in C(fn, p, m_, f, wd)])

    def spread(fn, p, m_, fracs, wd):
        """(tgen_spread, tident_spread, used_fracs, floor_fracs)."""
        used, floor = [], []
        for f in fracs:
            rs = C(fn, p, m_, f, wd)
            (used if rs and gen_ok(rs) else floor).append(f)
        tgs = [M(fn, p, m_, f, wd, "t_gen") for f in used]
        tis = [M(fn, p, m_, f, wd, "t_ident") for f in used]
        tgs = [x for x in tgs if x]
        tis = [x for x in tis if x]
        if len(tgs) < 2 or len(tis) < 2:
            return None, None, used, floor
        return max(tgs) / min(tgs), max(tis) / min(tis), used, floor

    V = {}
    gs, is_, used, floor = spread("add", 113, 512, A_FRACS, 1.0)
    V["P-H1"] = {"tgen_spread": gs, "tident_spread": is_, "fracs": used,
                 "data_floor": floor,
                 "result": ("KILL" if is_ and is_ >= 5 else
                            "PASS" if gs and gs >= 20 and is_ < 3
                            else "FAIL/partial")}

    d2 = {f: (None if M("add", 113, 512, f, 1.0, "t_ident") is None
              or M("add", 113, 512, f, 3.0, "t_ident") is None
              else abs(M("add", 113, 512, f, 1.0, "t_ident")
                       - M("add", 113, 512, f, 3.0, "t_ident")))
          for f in A_FRACS}
    tg1 = M("add", 113, 512, 0.3, 1.0, "t_gen")
    tg3 = M("add", 113, 512, 0.3, 3.0, "t_gen")
    V["P-H2"] = {"abs_ti_diff_by_frac": d2,
                 "tgen_ratio_f0.3": tg1 / tg3 if tg1 and tg3 else None,
                 "result": "PASS" if all(d is not None and d <= 25
                                         for d in d2.values())
                 and tg1 and tg3 and tg1 / tg3 >= 2 else "FAIL/partial"}

    def h34(fn, p, fracs, name):
        gs, is_, used, floor = spread(fn, p, 512, fracs, 1.0)
        dd = {f: (None if M(fn, p, 512, f, 1.0, "t_ident") is None
                  or M(fn, p, 512, f, 3.0, "t_ident") is None
                  else abs(M(fn, p, 512, f, 1.0, "t_ident")
                           - M(fn, p, 512, f, 3.0, "t_ident")))
              for f in fracs}
        ok = (gs and gs >= 5 and is_ and is_ < 3
              and all(d is not None and d <= 25 for d in dd.values()))
        return {"tgen_spread": gs, "tident_spread": is_, "data_floor": floor,
                "abs_ti_diff_by_frac": dd,
                "result": "PASS" if ok else "FAIL/partial"}

    V["P-H3"] = h34("sub", 113, B_FRACS, "sub113")
    p96 = h34("add", 96, B_FRACS, "add96")
    d127 = {f: (None if M("add", 127, 512, f, 1.0, "t_ident") is None
                or M("add", 127, 512, f, 3.0, "t_ident") is None
                else abs(M("add", 127, 512, f, 1.0, "t_ident")
                         - M("add", 127, 512, f, 3.0, "t_ident")))
            for f in (0.6, 0.9)}
    V["P-H4"] = {"p96": p96, "p127_abs_ti_diff": d127,
                 "result": "PASS" if p96["result"] == "PASS"
                 and all(d is not None and d <= 25 for d in d127.values())
                 else "FAIL/partial"}

    h5 = {}
    for m_ in (128, 2048):
        tg = [M("add", 113, m_, f, 3.0, "t_gen") for f in (0.3, 0.9)]
        ti = [M("add", 113, m_, f, 3.0, "t_ident") for f in (0.3, 0.9)]
        h5[m_] = {"t_gen": tg, "t_ident": ti,
                  "ok": all(tg) and all(ti) and tg[0] / tg[1] >= 5
                  and max(ti) / min(ti) <= 3}
    V["P-H5"] = {"by_width": h5,
                 "result": "PASS" if all(v["ok"] for v in h5.values())
                 else "FAIL/partial"}

    h6 = {}
    for wd in (1.0, 2.0):
        gs, is_, used, floor = spread("s5", 120, 512, [0.6, 0.7, 0.8, 0.9], wd)
        h6[wd] = {"tgen_spread": gs, "tident_spread": is_,
                  "fracs_used": used, "data_floor": floor,
                  "ok": gs is not None and len(used) >= 2
                  and gs >= 3 and is_ <= 3}
    scorable = any(len(v["fracs_used"]) >= 2 for v in h6.values())
    V["P-H6"] = {"by_wd": h6,
                 "result": ("NOT SCORED (fewer than 2 generalizing fracs)"
                            if not scorable else
                            "PASS" if any(v["ok"] for v in h6.values())
                            else "FAIL/partial")}

    bad = {f"{k}": float(np.mean([r["tilt_auc0"] for r in rs]))
           for k, rs in cells.items() if k[0] != "s5"
           and np.mean([r["tilt_auc0"] for r in rs]) < 0.60}
    V["P-H7"] = {"cells_below_0.60": bad,
                 "result": "PASS" if not bad else "FAIL/partial"}

    print()
    for k, v in V.items():
        print(f"{k}: {json.dumps(v, default=str)}")

    print("\nexploratory cross-frac committee J (add113 vs frac 0.3, wd 1):")
    base = {(r["ds"], r["is"]): r["committee"]
            for r in C("add", 113, 512, 0.3, 1.0)}
    for f in A_FRACS:
        if f == 0.3:
            continue
        js = [jaccard(r["committee"], base[(r["ds"], r["is"])])
              for r in C("add", 113, 512, f, 1.0)
              if (r["ds"], r["is"]) in base]
        if js:
            print(f"  frac {f}: median J {np.median(js):.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"runs": rows, "verdicts": V}, indent=1,
                              default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
