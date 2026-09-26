"""Programme predictions for a list of (p, s) (runs ON A POD, one process per (p, s), 1 thread each).

Usage: python experiments/common/predict.py OUT_DIR p:s [p:s ...]      (parallel: one process per pair)
Writes OUT_DIR/pred_p{p}_s{s}.json with gamma, K, K_all, y (canonical), y_all (every optimal vertex and all
dilates), tight offsets, the certification label from core.programme.solve_any ('exact (vertex cache)',
'exact (certified K cap)', 'best-found (not certified)'), and, beyond the cache, the best-found details and
K cap. For p in the vertex cache, a best-found cross-check is added when --crosscheck is given.
"""
import json, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
torch.set_num_threads(1)
from core import programme as P


def one(out_dir, p, s, crosscheck=False):
    t0 = time.time()
    r = P.solve_any(p, s)
    y_all = np.asarray(r["y_all"])
    rec = {"p": p, "s": s, "gamma": r["gamma"], "K": r["K"], "K_all": r["K_all"], "label": r["label"],
           "y": np.asarray(r["y"]).tolist(), "y_all": y_all.tolist(), "n_optimal_rows": int(len(y_all)),
           "tight_offsets": [P.tight_offsets(y, p).tolist() for y in y_all[:1]],
           "seconds": time.time() - t0, "programme_sha256": P.SELF_SHA256 if hasattr(P, "SELF_SHA256") else None}
    if "kcap" in r:
        rec["kcap"] = {k: (float(v) if v is not None else None) for k, v in r["kcap"].items()}
    if "best_found" in r:
        bf = r["best_found"]
        rec["best_found"] = {k: bf[k] for k in ("gamma", "K", "support", "n_local_minima", "distinct", "n_starts", "seed", "seconds")}
    if "best_found_gamma" in r:
        rec["best_found_gamma"] = r["best_found_gamma"]
    if crosscheck and r["label"].startswith("exact (vertex"):
        bf = P.best_found(p, s)
        rec["crosscheck_best_found"] = {"gamma": bf["gamma"], "K": bf["K"], "rel_gap": 1 - bf["gamma"] / r["gamma"]}
    fn = os.path.join(out_dir, f"pred_p{p}_s{s:g}.json")
    with open(fn + ".tmp", "w") as f:
        json.dump(rec, f)
    os.replace(fn + ".tmp", fn)
    print(f"p={p} s={s} K={rec['K']} gamma={rec['gamma']:.8f} {rec['label']} {rec['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    out = sys.argv[1]
    cc = "--crosscheck" in sys.argv
    os.makedirs(out, exist_ok=True)
    for a in sys.argv[2:]:
        if a.startswith("--"):
            continue
        p, s = a.split(":")
        one(out, int(p), float(s), cc)
