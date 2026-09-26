"""Summarise E0 tuning runs: convergence diagnostics only (NO agreement with predictions, NORMS C14).

Usage (from heracles-emp/): uv run python experiments/E0_tuning/analysis/summarize.py [ROUND_PREFIX]
Prints one line per run: cell, hyperparameters, status, final grad_rel, loss and gamma relative changes over
the final window, final gamma, wall time; and per-cell convergence fractions.
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(prefix=""):
    rows = {}
    with open(os.path.join(HERE, "RUNS.csv")) as f:
        for r in csv.DictReader(f):
            rows[r["run_id"]] = r
    out = []
    for rid, r in sorted(rows.items()):
        if not rid.startswith(prefix):
            continue
        d = os.path.join(HERE, "runs", rid)
        rec = {"run_id": rid, "status": r["status"]}
        try:
            conf = json.load(open(os.path.join(d, "config.json")))
            met = json.load(open(os.path.join(d, "metrics.json")))
        except FileNotFoundError:
            out.append(rec)
            continue
        cv = met.get("convergence", {}).get("values", {})
        rec.update({k: conf.get(k) for k in ("model", "nu", "p", "q", "width", "lam", "lr", "momentum", "steps",
                                             "init_scale", "lam_schedule", "lr_schedule")})
        rec.update({"grad_rel": cv.get("grad_rel"), "loss_rel_change": cv.get("loss_rel_change"),
                    "gamma_rel_change": cv.get("gamma_rel_change"), "margin": cv.get("margin"),
                    "gamma": met.get("measurement", {}).get("gamma"), "wall": conf.get("wall_seconds_batch"),
                    "checks": met.get("convergence", {}).get("checks")})
        out.append(rec)
    return out


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else ""
    recs = load(prefix)
    cells = {}
    for r in recs:
        cell = "_".join(r["run_id"].split("_")[:-1])
        cells.setdefault(cell, []).append(r)
        if "grad_rel" in r and all(r.get(k) is not None for k in ("grad_rel", "loss_rel_change", "gamma_rel_change", "gamma")):
            fl = [k for k, v in (r.get("checks") or {}).items() if not v]
            print(f"{r['run_id']:34s} {r['status']:13s} lr={r['lr']:<7g} mom={r['momentum']:<4g} lam={r['lam']:<8g} "
                  f"q={r['q']:<4g} init={r['init_scale']:<6g} T={r['steps']:<7d} gr={r['grad_rel']:.1e} "
                  f"dL={r['loss_rel_change']:.1e} dg={r['gamma_rel_change']:.1e} g={r['gamma']:.4e} "
                  f"wall={r['wall']:.0f}s fail={fl}")
        else:
            print(f"{r['run_id']:34s} {r['status']:13s} lr={r.get('lr')} mom={r.get('momentum')} lam={r.get('lam')} "
                  f"init={r.get('init_scale')} gr={r.get('grad_rel')}")
    print()
    for cell, rs in cells.items():
        ok = sum(r["status"] == "ok" for r in rs)
        print(f"{cell:30s} ok {ok}/{len(rs)}")


if __name__ == "__main__":
    main()
