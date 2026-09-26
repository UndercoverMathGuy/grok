"""Training: mean CE over the full table + lambda * sum_i ||omega_i||^q, GD or SGD(+momentum) only.

NORMS A4-A6, B8-B10. One process trains a BATCH of B >= 1 independent runs of the same shape in
lockstep (same model, n, p, width, steps, dtype, device, optimiser kind, log interval); per run they may
differ in q, lambda (+ schedule), lr (+ schedule), momentum, init scale and seed. Runs never interact:
the loss is a sum of per-run losses, so every run receives exactly its own gradient.

Precision: training dtype float32 (default) with TF32 disabled and float32 matmul precision 'highest'
(asserted at startup and recorded in config.json); every logged margin / normalised margin and every
measurement is computed in float64 from weights cast to float64.

Each run leaves, in its own directory (immutable once finished; files are made read-only):
    config.json   every hyperparameter, schedules, init, seed, dtype/device/precision flags, package
                  versions, sha256 of every core/*.py file, hostname, start/end time, batch composition
    log.csv       step, lr, lam, ce, ce64, penalty, total, total64, grad_norm, grad_rel, margin, gamma, acc
    final.pt      final weights {U, W} in the training dtype (+ the float64 cast is recomputed on load)
    metrics.json  status, convergence verdict, and core.measure.full_measurement (float64); agreement
                  metrics only when the config asks for them (never during tuning, NORMS C14)
Usage:  python -m core.train BATCH.json      (BATCH.json = {"runs": [{"out_dir": ..., "config": {...}}, ...]})
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import socket
import stat
import sys
import time

import numpy as np
import torch

from . import models as M

CORE_DIR = os.path.dirname(os.path.abspath(__file__))

SHAPE_KEYS = ("model", "n", "p", "width", "steps", "dtype", "device", "optimizer", "log_every", "batch_size")
PER_RUN_KEYS = ("q", "lam", "lam_schedule", "lr", "lr_schedule", "momentum", "nesterov", "init_scale", "init_scheme",
                "seed", "init_relabel_u")

DEFAULTS = {
    "model": "product",
    "optimizer": "gd",          # 'gd' (full batch) or 'sgd' (minibatch rows of the table)
    "momentum": 0.0,            # heavy-ball momentum (PyTorch SGD convention); 0 = plain GD/SGD
    "nesterov": False,
    "batch_size": None,         # SGD only: number of table rows per step
    "lam_schedule": {"kind": "constant"},
    "lr_schedule": {"kind": "constant"},
    "init_scheme": "normal",
    "init_scale": 1.0,
    "init_relabel_u": 1,        # E6: token relabelling x -> u x of the init: U'[.., a] = U[.., u^{-1} a mod p] (1 = none)
    "dtype": "float32",
    "device": "cpu",
    "log_every": 1000,
    "convergence": None,        # dict of thresholds, see convergence_verdict
    "max_seconds": 7200.0,      # hard wall-clock cap per batch (NORMS F22: every run <= 2 h)
    "measure_ablation": True,
    "score_predictions": False, # agreement metrics vs core.programme (never in tuning)
}

ALLOWED_OPTIMIZERS = ("gd", "sgd")


# ----------------------------------------------------------------------------------------------
# provenance
# ----------------------------------------------------------------------------------------------

def code_hashes() -> dict:
    """sha256 of every core/*.py file (the code this process imported)."""
    out = {}
    for fn in sorted(os.listdir(CORE_DIR)):
        if fn.endswith(".py"):
            with open(os.path.join(CORE_DIR, fn), "rb") as f:
                out[f"core/{fn}"] = hashlib.sha256(f.read()).hexdigest()
    return out


def set_precision_flags():
    """NORMS A6: TF32 off everywhere, float32 matmul precision 'highest'. Asserted and returned."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    flags = {
        "torch.backends.cuda.matmul.allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "torch.backends.cudnn.allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    assert flags["torch.backends.cuda.matmul.allow_tf32"] is False
    assert flags["torch.backends.cudnn.allow_tf32"] is False
    assert flags["float32_matmul_precision"] == "highest"
    return flags


def environment() -> dict:
    import scipy
    env = {
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "torch_num_threads": torch.get_num_threads(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        env["cuda_device"] = torch.cuda.get_device_name(0)
        env["cuda_version"] = torch.version.cuda
    return env


# ----------------------------------------------------------------------------------------------
# schedules
# ----------------------------------------------------------------------------------------------

def lr_at(cfg: dict, t: int) -> float:
    """Declared LR schedules: constant | cosine (to min_frac * lr) | warmup (linear over `warmup` steps)
    optionally combined: {"kind": "cosine", "min_frac": 0.1, "warmup": 100}."""
    sch = cfg.get("lr_schedule") or {"kind": "constant"}
    lr = float(cfg["lr"])
    T = int(cfg["steps"])
    f = 1.0
    w = int(sch.get("warmup", 0))
    if w > 0 and t < w:
        f *= (t + 1) / w
    kind = sch["kind"]
    if kind == "constant":
        pass
    elif kind == "cosine":
        mf = float(sch.get("min_frac", 0.0))
        f *= mf + (1 - mf) * 0.5 * (1 + math.cos(math.pi * min(t / T, 1.0)))
    elif kind == "warmup":
        pass
    else:
        raise ValueError(f"undeclared lr schedule {kind}")
    return lr * f


def lam_at(cfg: dict, t: int) -> float:
    """Declared lambda schedules: constant | geometric (lam0 -> lam over the first until_frac of steps,
    then constant lam)."""
    sch = cfg.get("lam_schedule") or {"kind": "constant"}
    lam = float(cfg["lam"])
    kind = sch["kind"]
    if kind == "constant":
        return lam
    if kind == "geometric":
        lam0 = float(sch["lam0"])
        T1 = float(sch.get("until_frac", 0.5)) * int(cfg["steps"])
        x = min(t / T1, 1.0) if T1 > 0 else 1.0
        return lam0 * (lam / lam0) ** x
    raise ValueError(f"undeclared lambda schedule {kind}")


# ----------------------------------------------------------------------------------------------
# convergence
# ----------------------------------------------------------------------------------------------

DEFAULT_CONVERGENCE = {
    "window_frac": 0.1,     # compare the final log row with the row window_frac * steps earlier
    "grad_rel_max": 1e-3,   # stationarity: ||g_CE + lam g_R|| / (||g_CE|| + lam ||g_R||) at the end
    "loss_rel_change_max": 1e-4,   # |L64(T) - L64(T - window)| / L64(T)
    "gamma_rel_change_max": 1e-3,  # |gamma(T) - gamma(T - window)| / |gamma(T)|
    "require_positive_margin": True,
}


def convergence_verdict(rows: list[dict], cfg: dict) -> dict:
    """Apply the declared convergence criterion to a run's log rows (see DEFAULT_CONVERGENCE).

    The comparison window starts no earlier than the end of any lambda anneal, so a run is only judged
    at its final lambda. Returns {'converged': bool, 'checks': {...}, 'values': {...}}.
    """
    crit = dict(DEFAULT_CONVERGENCE)
    crit.update(cfg.get("convergence") or {})
    T = int(cfg["steps"])
    last = rows[-1]
    t_ref = T - crit["window_frac"] * T
    sch = cfg.get("lam_schedule") or {"kind": "constant"}
    if sch["kind"] == "geometric":
        t_anneal_end = float(sch.get("until_frac", 0.5)) * T
        if t_ref < t_anneal_end:
            t_ref = t_anneal_end
    ref = min(rows, key=lambda r: abs(r["step"] - t_ref))
    vals = {
        "grad_rel": last["grad_rel"],
        "loss_rel_change": abs(last["total64"] - ref["total64"]) / abs(last["total64"]) if last["total64"] else float("inf"),
        "gamma_rel_change": abs(last["gamma"] - ref["gamma"]) / abs(last["gamma"]) if last["gamma"] else float("inf"),
        "margin": last["margin"],
        "ref_step": ref["step"],
        "final_step": last["step"],
    }
    finite = all(np.isfinite(v) for k, v in vals.items() if k not in ("ref_step", "final_step"))
    checks = {
        "finite": bool(finite),
        "reached_final_step": last["step"] >= T,
        "grad_rel": bool(vals["grad_rel"] <= crit["grad_rel_max"]),
        "loss_plateau": bool(vals["loss_rel_change"] <= crit["loss_rel_change_max"]),
        "gamma_stable": bool(vals["gamma_rel_change"] <= crit["gamma_rel_change_max"]),
        "positive_margin": bool(vals["margin"] > 0) if crit["require_positive_margin"] else True,
        "window_after_anneal": ref["step"] < last["step"],
    }
    return {"converged": all(checks.values()), "checks": checks, "values": vals, "criterion": crit}


# ----------------------------------------------------------------------------------------------
# the batched trainer
# ----------------------------------------------------------------------------------------------

def _dtype(name: str):
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _make_readonly(path: str):
    for fn in os.listdir(path):
        fp = os.path.join(path, fn)
        if os.path.isfile(fp):
            os.chmod(fp, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def validate_batch(runs: list[dict]) -> list[dict]:
    cfgs = []
    for r in runs:
        c = dict(DEFAULTS)
        c.update(r["config"])
        for k in ("model", "n", "p", "q", "width", "lam", "lr", "steps", "seed"):
            if k not in c:
                raise ValueError(f"config missing {k}")
        if c["optimizer"] not in ALLOWED_OPTIMIZERS:
            raise ValueError(f"optimizer {c['optimizer']} is not allowed (NORMS A5)")
        if c["model"] not in M.MODELS:
            raise ValueError(c["model"])
        nu = c["n"] + 1
        if not (2.0 <= float(c["q"]) <= nu + 1e-12):
            raise ValueError(f"q={c['q']} outside [2, nu={nu}] (NORMS A4)")
        if c["optimizer"] == "gd" and c.get("batch_size"):
            raise ValueError("gd is full batch; use optimizer 'sgd' for minibatches")
        c["nu"] = nu
        c["s"] = nu / float(c["q"])
        cfgs.append(c)
    for k in SHAPE_KEYS:
        vals = {json.dumps(c.get(k)) for c in cfgs}
        if len(vals) != 1:
            raise ValueError(f"batched runs must share {k}: {vals}")
    return cfgs


def train_batch(runs: list[dict], verbose: bool = True) -> list[dict]:
    """Train B runs in lockstep. runs = [{"out_dir": str, "run_id": str, "config": {...}}, ...].

    Returns one summary dict per run (status, convergence verdict, final gamma).
    """
    t_start = time.time()
    start_str = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    flags = set_precision_flags()
    cfgs = validate_batch(runs)
    c0 = cfgs[0]
    device = torch.device(c0["device"])
    if device.type == "cpu":
        torch.set_num_threads(int(os.environ.get("HERACLES_THREADS", "1")))
    dt = _dtype(c0["dtype"])
    B = len(cfgs)
    model, n, p, m, T = c0["model"], int(c0["n"]), int(c0["p"]), int(c0["width"]), int(c0["steps"])
    nu = n + 1
    log_every = int(c0["log_every"])
    for r in runs:
        od = r["out_dir"]
        if os.path.exists(od) and os.listdir(od):
            raise FileExistsError(f"run dir {od} is not empty: runs are immutable (NORMS B8/B9)")
        os.makedirs(od, exist_ok=True)

    hashes = code_hashes()
    env = environment()
    batch_ids = [r.get("run_id", os.path.basename(r["out_dir"].rstrip("/"))) for r in runs]
    for r, c in zip(runs, cfgs):
        conf = {
            "run_id": r.get("run_id", os.path.basename(r["out_dir"].rstrip("/"))),
            **{k: c[k] for k in sorted(c)},
            "precision_flags": flags,
            "environment": env,
            "code_sha256": hashes,
            "batch": {"size": B, "run_ids": batch_ids},
            "start_time": start_str,
            "argv": sys.argv,
        }
        with open(os.path.join(r["out_dir"], "config.json"), "w") as f:
            json.dump(conf, f, indent=1, default=str)

    # --- init (each run from its own seed; float64 generated, float32-rounded) ---
    Us, Ws = [], []
    for c in cfgs:
        U0, W0 = M.init_params(model, n, m, p, int(c["seed"]), float(c["init_scale"]), c["init_scheme"])
        u = int(c.get("init_relabel_u", 1)) % p
        if u != 1:
            # relabel tokens a -> u a (inputs and outputs): new vector v'(a) = v(u^{-1} a mod p)
            if u == 0:
                raise ValueError("init_relabel_u must be a unit mod p")
            src = torch.tensor([(pow(u, -1, p) * a) % p for a in range(p)])
            U0, W0 = U0[..., src].clone(), W0[..., src].clone()
        Us.append(U0)
        Ws.append(W0)
    U = torch.cat(Us).to(device=device, dtype=dt).requires_grad_(True)
    W = torch.cat(Ws).to(device=device, dtype=dt).requires_grad_(True)
    y = M.labels(p, n, device=device)
    y64 = y
    q = torch.tensor([float(c["q"]) for c in cfgs], dtype=dt, device=device)
    q64 = q.double()
    mom = torch.tensor([float(c["momentum"]) for c in cfgs], dtype=dt, device=device)
    bufU = torch.zeros_like(U)
    bufW = torch.zeros_like(W)
    status = ["running"] * B
    N = p ** n
    sgd = c0["optimizer"] == "sgd"
    bs = int(c0["batch_size"]) if sgd and c0.get("batch_size") else None
    gens = [torch.Generator(device="cpu").manual_seed(int(c["seed"]) + 1_000_003) for c in cfgs] if sgd else None

    logs = [[] for _ in range(B)]
    writers = []
    files = []
    fields = ["step", "lr", "lam", "ce", "ce64", "penalty", "total", "total64", "grad_norm", "grad_rel", "margin",
              "gamma", "acc", "wall_s"]
    for r in runs:
        fh = open(os.path.join(r["out_dir"], "log.csv"), "w", newline="")
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        files.append(fh)
        writers.append(w)

    def loss_terms(Ux, Wx, rows_idx=None):
        F = M.logits(model, Ux, Wx)
        if rows_idx is None:
            ce = M.ce_margin_form(F, y)
        else:
            ce = torch.stack([M.ce_margin_form(F[b:b + 1, rows_idx[b]], y[rows_idx[b]])[0] for b in range(B)])
        pen = M.penalty(Ux, Wx, q)
        return ce, pen

    per_run_measure = B * (p ** (n + 1)) * 8 > 2e9  # float64 tables of all runs would exceed ~2 GB

    def measure64(step, lrs, lams, ce, pen, tot, gnorm, grel):
        with torch.no_grad():
            U64, W64 = U.detach().double(), W.detach().double()
            parts = [(U64, W64, q64)] if not per_run_measure else \
                [(U64[b:b + 1], W64[b:b + 1], q64[b:b + 1]) for b in range(B)]
            res = []
            for Ux, Wx, qx in parts:
                F64 = M.logits(model, Ux, Wx)
                res.append((M.ce_margin_form(F64, y64), M.penalty(Ux, Wx, qx), M.margins_from_logits(F64, y64),
                            (F64.argmax(dim=2) == y64.view(1, -1)).double().mean(dim=1)))
                del F64
            ce64, pen64, marg, acc = [torch.cat([r_[i] for r_ in res]).cpu() for i in range(4)]
            nrm = pen64 ** (1.0 / q64.cpu())
            gam = marg / nrm ** nu
        for b in range(B):
            row = {
                "step": step, "lr": lrs[b], "lam": lams[b], "ce": float(ce[b]), "ce64": float(ce64[b]),
                "penalty": float(pen64[b]), "total": float(tot[b]),
                "total64": float(ce64[b] + lams[b] * pen64[b]), "grad_norm": float(gnorm[b]),
                "grad_rel": float(grel[b]), "margin": float(marg[b]), "gamma": float(gam[b]),
                "acc": float(acc[b]), "wall_s": time.time() - t_start,
            }
            logs[b].append(row)
            writers[b].writerow(row)
            files[b].flush()

    # Performance: no per-step host<->device synchronisation. Schedules are precomputed in chunks and
    # indexed on the device; gradient statistics, float64 measurements and the divergence check run
    # only at log steps. (No torch.use_deterministic_algorithms: it would slow the GPU down.)
    use_mom = any(float(c["momentum"]) != 0 for c in cfgs)
    nest_t = torch.tensor([bool(c["nesterov"]) for c in cfgs], device=device)
    active_f = torch.ones(B, dtype=dt, device=device)
    CH = 4096
    sched_start = -CH
    lr_ch = lam_ch = None
    lr_host = lam_host = None
    t = 0
    timed_out = False
    while True:
        if t - sched_start >= CH:
            sched_start = t
            ts = range(t, min(t + CH, T + 1))
            lr_host = np.array([[lr_at(c, tt) for c in cfgs] for tt in ts])
            lam_host = np.array([[lam_at(c, tt) for c in cfgs] for tt in ts])
            lr_ch = torch.tensor(lr_host, dtype=dt, device=device)
            lam_ch = torch.tensor(lam_host, dtype=dt, device=device)
        k = t - sched_start
        lam_t = lam_ch[k]
        lr_t = lr_ch[k] * active_f
        rows_idx = None
        if sgd:
            rows_idx = [torch.randperm(N, generator=g)[:bs].to(device) for g in gens]
        ce, pen = loss_terms(U, W, rows_idx)
        tot = ce + lam_t * pen
        gU, gW = torch.autograd.grad(tot.sum(), (U, W))
        is_log = (t % log_every == 0) or (t == T)
        over_time = (time.time() - t_start) > float(c0["max_seconds"])
        if is_log or over_time:
            with torch.no_grad():
                # stationarity: g_R analytic, g_CE = g_tot - lam g_R (small tensors: parameters only)
                rU, rW = M.penalty_grad(U, W, q)
                l64 = lam_t.double()
                gU64, gW64, rU64, rW64 = gU.double(), gW.double(), rU.double(), rW.double()
                gnorm = torch.sqrt((gU64 ** 2).sum(dim=(1, 2, 3)) + (gW64 ** 2).sum(dim=(1, 2)))
                cU, cW = gU64 - l64[:, None, None, None] * rU64, gW64 - l64[:, None, None] * rW64
                gce = torch.sqrt((cU ** 2).sum(dim=(1, 2, 3)) + (cW ** 2).sum(dim=(1, 2)))
                gr = torch.sqrt((rU64 ** 2).sum(dim=(1, 2, 3)) + (rW64 ** 2).sum(dim=(1, 2)))
                grel = gnorm / (gce + l64 * gr)
                bad = (~torch.isfinite(tot.detach()) | ~torch.isfinite(gnorm)).cpu()
                for b in torch.nonzero(bad).flatten().tolist():
                    if status[b] == "running":
                        status[b] = "diverged"
                        active_f[b] = 0.0
            measure64(t, lr_host[k].tolist(), lam_host[k].tolist(), ce.detach().cpu(), pen.detach().cpu(),
                      tot.detach().cpu(), gnorm.cpu(), grel.cpu())
            if verbose and B <= 8:
                print(f"t={t} " + " | ".join(f"ce={logs[b][-1]['ce64']:.3e} g={logs[b][-1]['gamma']:.4e} "
                                              f"gr={logs[b][-1]['grad_rel']:.1e}" for b in range(B) if logs[b]),
                      flush=True)
            if all(st_ != "running" for st_ in status):
                break
        if t >= T:
            break
        if over_time:
            timed_out = True
            break
        with torch.no_grad():
            if use_mom:
                bufU.mul_(mom[:, None, None, None]).add_(gU)
                bufW.mul_(mom[:, None, None]).add_(gW)
                dU = torch.where(nest_t[:, None, None, None], gU + mom[:, None, None, None] * bufU, bufU)
                dW = torch.where(nest_t[:, None, None], gW + mom[:, None, None] * bufW, bufW)
            else:
                dU, dW = gU, gW
            U.sub_(lr_t[:, None, None, None] * dU)
            W.sub_(lr_t[:, None, None] * dW)
        t += 1

    for fh in files:
        fh.close()
    end_str = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    wall = time.time() - t_start
    summaries = []
    from . import measure as MS
    for b, (r, c) in enumerate(zip(runs, cfgs)):
        od = r["out_dir"]
        Ub, Wb = U.detach()[b:b + 1].cpu(), W.detach()[b:b + 1].cpu()
        torch.save({"U": Ub.clone(), "W": Wb.clone(), "dtype": c["dtype"], "step": logs[b][-1]["step"] if logs[b] else 0},
                   os.path.join(od, "final.pt"))
        if status[b] == "running":
            status[b] = "finished"
        verdict = convergence_verdict(logs[b], c) if logs[b] else {"converged": False}
        if status[b] == "diverged":
            final_status = "diverged"
        elif timed_out:
            final_status = "not_converged"
        else:
            final_status = "ok" if verdict["converged"] else "not_converged"
        met = {"status": final_status, "timed_out": timed_out, "convergence": verdict, "wall_seconds_batch": wall,
               "end_time": end_str}
        if final_status != "diverged":
            try:
                met["measurement"] = MS.full_measurement(model, Ub, Wb, float(c["q"]),
                                                         with_ablation=bool(c["measure_ablation"]))
            except Exception as e:  # recorded, never silently dropped
                met["measurement_error"] = repr(e)
                met["status"] = final_status = "crashed"
            if c.get("score_predictions"):
                from . import metrics as MT
                met["agreement"] = MT.score_run(met["measurement"])
        with open(os.path.join(od, "metrics.json"), "w") as f:
            json.dump(met, f, indent=1, default=_json_default)
        # append end time to config (config.json is rewritten once, before it is made read-only)
        cp = os.path.join(od, "config.json")
        with open(cp) as f:
            conf = json.load(f)
        conf["end_time"] = end_str
        conf["wall_seconds_batch"] = wall
        with open(cp, "w") as f:
            json.dump(conf, f, indent=1, default=str)
        _make_readonly(od)
        last = logs[b][-1] if logs[b] else {}
        summaries.append({"run_id": r.get("run_id"), "out_dir": od, "status": final_status,
                          "gamma": last.get("gamma"), "grad_rel": last.get("grad_rel"),
                          "ce64": last.get("ce64"), "wall_s": wall})
    return summaries


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, complex):
        return [o.real, o.imag]
    return str(o)


def load_final(run_dir: str):
    """(U, W) of a finished run as float64 tensors of shape (1, n, m, p), (1, m, p)."""
    d = torch.load(os.path.join(run_dir, "final.pt"), map_location="cpu")
    return d["U"].double(), d["W"].double()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    with open(argv[0]) as f:
        spec = json.load(f)
    summ = train_batch(spec["runs"])
    print(json.dumps(summ, default=_json_default))
    return 0


if __name__ == "__main__":
    sys.exit(main())
