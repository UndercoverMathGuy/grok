"""Predictability-onset curve: how early in training is grok time readable?
Sweep forecast epoch FC backwards; at each FC, featurize every run from
data <= FC only, fit ridge (leave-cell-out), report held-out skill on
log10 t_gen. Fixed population: add-113 runs with t_gen > 500 and spectra
grid <= 25 ep. Law features included where memorization has happened.
"""
import json
import glob
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RS = 10 ** -2.25
FCS = [25, 50, 100, 150, 200, 300, 400, 500]


def fs(m):
    i = np.where(m[:-1] & m[1:])[0]
    return int(i[0]) if len(i) else -1


def load_runs():
    rows = json.load(open(ROOT / "notes" / "v3_features.json"))
    keep = []
    pats = {"degrok": "runs_degrok", "degrok2": "runs_degrok2",
            "clock": "runs_clock", "t2c": "runs_flight/t2c",
            "t2d": "runs_flight/t2d", "t2dv2": "runs_flight/t2dv2",
            "t2e": "runs_flight/t2e", "floor": "runs_flight/floornoise"}
    for r in rows:
        if r["_cohort"] not in pats:
            continue                                   # drop 100-ep grids
        rd = ROOT / pats[r["_cohort"]] / "/".join(r["_run"].split("/")[-2:])
        if not (rd / "spectra.npz").exists():
            rd = ROOT / r["_run"]
            if not (rd / "spectra.npz").exists():
                continue
        z = np.load(rd / "spectra.npz")
        ep = z["epochs"]
        if len(ep) > 1 and ep[1] - ep[0] > 25:
            continue
        if (rd / "seg2" / "spectra.npz").exists():
            z2 = np.load(rd / "seg2" / "spectra.npz")
            cat = lambda k: np.concatenate([z[k], z2[k]])
            ep = np.concatenate([ep, 500 + z2["epochs"]])
            E, C = cat("energy"), cat("coeffs")
            ta, te = cat("train_acc"), cat("test_acc")
        else:
            E, C, ta, te = z["energy"], z["coeffs"], z["train_acc"], z["test_acc"]
        ig = fs(te >= 0.9)
        if ig < 0 or ep[ig] <= 500:
            continue
        keep.append(dict(ep=ep, E=E, C=C, ta=ta,
                         y=float(np.log10(ep[ig])),
                         cell=r["_cohort"] + "/" + r["_cell"]))
    return keep


def feats_at(run, fc):
    ep, E, C, ta = run["ep"], run["E"], run["C"], run["ta"]
    lim = np.searchsorted(ep, fc, "right")
    if lim < 3:
        return None
    ep, E, C, ta = ep[:lim], E[:lim], C[:lim], ta[:lim]
    Efc = E[-1]
    order = np.argsort(Efc)[::-1]
    comm = order[:4].tolist()
    junk = [j for j in range(E.shape[1]) if j not in comm]
    r = E[:, junk].sum(1) / E.sum(1)
    le = np.log10(np.clip(Efc, 1e-12, None))
    so = le[order]
    Ej = Efc[junk]
    pj = Ej / max(Ej.sum(), 1e-12)
    coh = np.abs(C[-1]) / np.sqrt(np.clip(Efc, 1e-12, None))
    w = np.arange(max(0, lim - 12), lim)      # trailing window for rates
    x = ep[w].astype(float)
    lg = lambda v: np.log(np.clip(v, 1e-12, None))
    g_tot = float(np.polyfit(x, lg(E[w].sum(1)), 1)[0]) * 1e3
    g_c = float(np.polyfit(x, lg(E[w][:, comm].sum(1)), 1)[0]) * 1e3
    lam = -float(np.polyfit(x, lg(r[w]), 1)[0]) * 1e3
    f = dict(log_r=float(np.log10(max(r[-1], 1e-12))),
             comm_hhi=float(((Efc[comm] / Efc[comm].sum()) ** 2).sum()),
             gap12=float(so[0] - so[1]), gap45=float(so[3] - so[4]),
             junk_max=float(Ej.max() / max(Efc.sum(), 1e-12)),
             junk_pr=float(1.0 / (pj ** 2).sum()),
             coh_c=float(coh[comm].mean()), coh_j=float(coh[junk].mean()),
             g_tot=g_tot, g_c=g_c, lam=lam,
             tracc=float(ta[-1]), log_e=float(np.log10(E[-1].sum())))
    im = fs(ta >= 0.99)
    f["memorized"] = float(im >= 0)
    if im >= 0:
        t_mem, r_mem = float(ep[im]), float(r[im])
        wl = np.where(ep >= t_mem)[0]
        laml = (-float(np.polyfit(ep[wl].astype(float), lg(r[wl]), 1)[0])
                if len(wl) >= 4 else 0.0)
        t_law = (t_mem + 132 if r_mem <= np.e * RS else
                 (t_mem + np.log(r_mem / RS) / laml if laml > 0 else 30000))
        f.update(t_mem=t_mem, log_r_mem=float(np.log10(max(r_mem, 1e-12))),
                 log_t_law=float(np.log10(t_law)))
    else:
        f.update(t_mem=np.nan, log_r_mem=np.nan, log_t_law=np.nan)
    return f


def main():
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    runs = load_runs()
    print(f"{len(runs)} runs, fixed population, t_gen > 500")
    names = None
    print(f"{'FC':>4} {'n':>4} {'mem%':>5} {'R2(log t_gen)':>13} "
          f"{'median |err| dex':>16} {'x-err':>6}")
    for fc in FCS:
        X, y, g = [], [], []
        for run in runs:
            f = feats_at(run, fc)
            if f is None:
                continue
            if names is None:
                names = sorted(f)
            X.append([f[k] for k in names])
            y.append(run["y"])
            g.append(run["cell"])
        X = np.array(X, float)
        y = np.array(y)
        g = np.array(g)
        for j in range(X.shape[1]):
            col = X[:, j]
            fin = np.isfinite(col)
            col[~fin] = np.median(col[fin]) if fin.any() else 0.0
        gkf = GroupKFold(n_splits=8)
        oof = np.full(len(y), np.nan)
        m = make_pipeline(StandardScaler(),
                          RidgeCV(alphas=np.logspace(-3, 3, 13)))
        for tr, te in gkf.split(X, y, g):
            m.fit(X[tr], y[tr])
            oof[te] = np.clip(m.predict(X[te]), np.log10(550), np.log10(40000))
        e = oof - y
        r2 = 1 - np.nanvar(e) / y.var()
        memp = 100 * np.mean(X[:, names.index("memorized")])
        med = np.nanmedian(np.abs(e))
        print(f"{fc:>4} {len(y):>4} {memp:>4.0f}% {r2:>13.3f} "
              f"{med:>16.3f} x{10**med:.2f}")


if __name__ == "__main__":
    main()
