"""Feature extraction for the v3 observable hunt.

For every generalized add-113 run on disk, compute forecast-time features
(spectra/metrics up to FC=500 only) plus the law's own prediction, and the
residual target y = log10(t_gen) - log10(t_law). Writes
notes/v3_features.json. Modeling lives in flight/v3_hunt.py.
"""
import json
import glob
import re
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FC = 500
RS = 10 ** -2.25
P = 113


def fs(m):
    i = np.where(m[:-1] & m[1:])[0]
    return int(i[0]) if len(i) else -1


def stitched(rd):
    z1 = np.load(rd / "spectra.npz")
    if not (rd / "seg2" / "spectra.npz").exists():
        return z1["epochs"], z1["energy"], z1["coeffs"], z1["train_acc"], z1["test_acc"]
    z2 = np.load(rd / "seg2" / "spectra.npz")
    cat = lambda k: np.concatenate([z1[k], z2[k]])
    return (np.concatenate([z1["epochs"], FC + z2["epochs"]]),
            cat("energy"), cat("coeffs"), cat("train_acc"), cat("test_acc"))


def harmonic_set(comm_freqs, p):
    h = set()
    for c in comm_freqs:
        k = (2 * c) % p
        h.add(min(k, p - k))
    return h


def features(rd, cfg):
    ep, E, C, ta, te = stitched(rd)
    im, ig = fs(ta >= 0.99), fs(te >= 0.9)
    if im < 0 or ig < 0:
        return None
    t_mem, t_gen = int(ep[im]), int(ep[ig])
    if t_gen <= FC:
        return None                       # forecastable runs only
    ifc = np.searchsorted(ep, FC, "right") - 1
    Efc = E[ifc]
    order = np.argsort(Efc)[::-1]
    comm = order[:4].tolist()             # K=4 pipeline (fixed)
    junk = [j for j in range(E.shape[1]) if j not in comm]
    r = E[:, junk].sum(1) / E.sum(1)
    r_mem = float(r[im])
    w = np.where((ep >= t_mem) & (ep <= FC))[0]
    if len(w) < 5:
        return None
    x, y = ep[w].astype(float), np.log(np.clip(r[w], 1e-12, None))
    lam = -float(np.polyfit(x, y, 1)[0])
    quad = float(np.polyfit(x, y, 2)[0])          # ramp curvature
    # law prediction (v2-style, no gamma)
    if r_mem <= np.e * RS:
        t_law = t_mem + 132
    elif lam <= 0:
        t_law = 30_000
    else:
        t_law = t_mem + max(0.0, np.log(r_mem / RS)) / lam
    lg = lambda v: np.log(np.clip(v, 1e-12, None))
    g_tot = float(np.polyfit(x, lg(E[w].sum(1)), 1)[0]) * 1e3
    g_c = float(np.polyfit(x, lg(E[w][:, comm].sum(1)), 1)[0]) * 1e3
    g_j = float(np.polyfit(x, lg(E[w][:, junk].sum(1)), 1)[0]) * 1e3
    le = np.log10(np.clip(Efc, 1e-12, None))
    so = le[order]
    Ej = Efc[junk]
    pj = Ej / max(Ej.sum(), 1e-12)
    comm_freqs = [c + 1 for c in comm]
    harm = harmonic_set(comm_freqs, P)
    hmask = np.array([(j + 1) in harm for j in junk])
    coh = np.abs(C[ifc]) / np.sqrt(np.clip(Efc, 1e-12, None))
    # train/test shape
    i50 = fs(ta >= 0.5)
    tl = json.loads((rd / "metrics.json").read_text())["test_losses"] \
        if (rd / "metrics.json").exists() else None
    f = dict(
        t_mem=t_mem, log_r_mem=float(np.log10(max(r_mem, 1e-12))),
        lam=lam * 1e3, quad=quad * 1e6, g_tot=g_tot, g_c=g_c, g_j=g_j,
        gap12=float(so[0] - so[1]), gap45=float(so[3] - so[4]),
        gap56=float(so[4] - so[5]),
        comm_hhi=float(((Efc[comm] / Efc[comm].sum()) ** 2).sum()),
        junk_pr=float(1.0 / (pj ** 2).sum()),
        junk_max_share=float(Ej.max() / max(Efc.sum(), 1e-12)),
        harm_frac=float(Ej[hmask].sum() / max(Ej.sum(), 1e-12)),
        coh_comm=float(coh[comm].mean()), coh_junk=float(coh[junk].mean()),
        t_50=int(ep[i50]) if i50 >= 0 else t_mem,
        log_t_law=float(np.log10(t_law)),
    )
    if tl is not None and len(tl) >= 6:
        f["test_loss_fc"] = float(np.log10(max(tl[5], 1e-12)))
        f["test_loss_slope"] = float(tl[5] - tl[2])
    f["_y"] = float(np.log10(t_gen) - np.log10(t_law))
    f["_log_t_gen"] = float(np.log10(t_gen))
    f["_cfg"] = dict(frac=cfg.get("frac_train"), wd=cfg.get("weight_decay"),
                     lr=cfg.get("lr"), width=cfg.get("d_mlp"))
    return f


rows = []
SETS = [
    ("degrok", f"{ROOT}/runs_degrok/p113_*/*"),
    ("degrok2", f"{ROOT}/runs_degrok2/add113_*/*"),
    ("dial", f"{ROOT}/runs_dial/*/*"),
    ("race", f"{ROOT}/runs_race/*/*"),
    ("clock", f"{ROOT}/runs_clock/base/*"),
    ("t2c", f"{ROOT}/runs_flight/t2c/*/*"),
    ("t2d", f"{ROOT}/runs_flight/t2d/*/*"),
    ("t2dv2", f"{ROOT}/runs_flight/t2dv2/*/*"),
    ("t2e", f"{ROOT}/runs_flight/t2e/add113_*/*"),
    ("floor", f"{ROOT}/runs_flight/floornoise/*/*"),
]
for cohort, pat in SETS:
    for d in sorted(glob.glob(pat)):
        rd = Path(d)
        if not (rd / "spectra.npz").exists():
            continue
        try:
            cfg = json.loads((rd / "config.json").read_text())
        except FileNotFoundError:
            m = re.search(r"wd([\d.]+)", d)
            cfg = {"weight_decay": float(m.group(1)) if m else None,
                   "lr": 1e-3, "frac_train": 0.3, "d_mlp": None}
        if cfg.get("p", 113) != 113 or cfg.get("fn_name", "add") != "add":
            continue
        f = features(rd, cfg)
        if f is None:
            continue
        f["_cohort"] = cohort
        f["_cell"] = d.split("/")[-2]
        f["_run"] = "/".join(d.split("/")[-3:])
        rows.append(f)
print(f"{len(rows)} runs featurized")
json.dump(rows, open(ROOT / "notes" / "v3_features.json", "w"), indent=0)
print("wrote notes/v3_features.json")
