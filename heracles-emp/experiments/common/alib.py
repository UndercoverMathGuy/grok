"""Shared analysis helpers (light; reads pulled run directories). Every metric comes from core.metrics / core.measure."""
import csv, glob, json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)
import numpy as np
from core import metrics as MT, measure as MS, programme as PG


def runs_status(exp):
    """Last status per run_id from RUNS.csv (append-only history)."""
    st = {}
    fn = os.path.join(ROOT, "experiments", exp, "RUNS.csv")
    if os.path.exists(fn):
        for r in csv.DictReader(open(fn)):
            st[r["run_id"]] = r
    return st


def load(exp):
    out = []
    st = runs_status(exp)
    for d in sorted(glob.glob(os.path.join(ROOT, "experiments", exp, "runs", "*"))):
        try:
            c = json.load(open(os.path.join(d, "config.json"))); m = json.load(open(os.path.join(d, "metrics.json")))
        except Exception as e:
            out.append({"run_id": os.path.basename(d), "status": "crashed", "error": repr(e)})
            continue
        out.append({"run_id": c["run_id"], "dir": d, "config": c, "metrics": m, "status": m["status"],
                    "csv_status": st.get(c["run_id"], {}).get("status")})
    return out


def pred_index(exp):
    P = json.load(open(os.path.join(ROOT, "experiments", exp, "predictions.json")))
    cells = P.get("cells") or [P["cell"]]
    idx = {}
    for c in cells:
        key = (c.get("model", "product"), c["nu"], c["p"], round(c["s"], 6))
        idx[key] = c
    return idx, P


def score(run, cell):
    """All registered agreement metrics of one run vs its prediction cell (core.metrics.score_run)."""
    m = run["metrics"]["measurement"]
    y_all = cell.get("y_all") or [cell["y_star"]]
    pred = {"gamma": cell["Gamma"], "y_all": np.asarray(y_all), "label": cell["label"], "K_all": cell.get("K_all", [cell["K_star"]])}
    return MT.score_run(m, pred)


def flat_row(run, cell, sc):
    c, m = run["config"], run["metrics"]["measurement"]
    conv = run["metrics"].get("convergence", {}).get("values", {})
    r = {"run_id": run["run_id"], "status": run["status"], "model": c["model"], "nu": c["n"] + 1, "p": c["p"], "q": c["q"],
         "s": round((c["n"] + 1) / c["q"], 6), "seed": c["seed"], "steps": c["steps"], "K_star": cell["K_star"], "label": cell["label"],
         "runner_up_rel_gap": (cell.get("runner_up") or {}).get("rel_gap"), "gamma": m["gamma"], "margin": m["margin"],
         "grad_rel": conv.get("grad_rel"), "gamma_rel_change": conv.get("gamma_rel_change"), "accuracy": m["accuracy"],
         "energy_clean": m["energy_frac"]["clean"], "energy_junk": m["energy_frac"]["junk"], "energy_offdiag": m["energy_frac"]["off_diag"]}
    for k, v in sc["counts"].items():
        r[f"cnt_{k}"] = v["meas"]; r[f"cnt_{k}_pred"] = v["pred_same_def"]; r[f"cnt_{k}_err"] = v["signed"]; r[f"cnt_{k}_exact"] = v["exact"]
    for k, v in sc["amp"].items():
        r[f"amp_{k}"] = v
    for k, v in sc["share"].items():
        r[f"share_{k}"] = v
    for t in MT.THRESHOLDS:
        r[f"jaccard_thr_{t}"] = sc[f"jaccard_thr_{t}"]
    for k, v in sc["gamma_over_ceiling"].items():
        r[f"gamma_over_{k}"] = v
    r["align_cos"] = sc["alignment"]["cosine"]; r["align_index"] = sc["alignment"]["index"]
    return r


def boot_ci(vals_by_seed, stat=np.median):
    b = MT.bootstrap_over_seeds(vals_by_seed, stat=stat, n_boot=2000)
    return b["stat"], b["ci"]


def write_csv(rows, fn):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "nan"
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.{nd}g}"
