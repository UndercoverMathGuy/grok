"""Shared v3 feature/forecast core. Used byte-identically by the training
extractor (flight/v3_features.py) and the pod forecast stage — any drift
between the two would silently invalidate the prereg, so both import THIS.
All computations use spectra data up to FC only."""
import json
import numpy as np

FC = 500
RS = 10 ** -2.25
P = 113
FEATURES = ["t_mem", "log_r_mem", "lam", "quad", "g_tot", "g_c", "g_j",
            "gap12", "gap45", "gap56", "comm_hhi", "junk_pr",
            "junk_max_share", "harm_frac", "coh_comm", "coh_junk",
            "t_50", "log_t_law", "test_loss_fc", "test_loss_slope"]


def fs(m):
    i = np.where(m[:-1] & m[1:])[0]
    return int(i[0]) if len(i) else -1


def harmonic_set(comm_freqs, p=P):
    h = set()
    for c in comm_freqs:
        k = (2 * c) % p
        h.add(min(k, p - k))
    return h


def core_features(ep, E, C, ta, te, test_losses=None):
    """Features from data <= FC. Returns (dict, status). status is 'ok',
    'no_mem' (no memorization by FC) or 'violated' (generalized by FC)."""
    lim = np.searchsorted(ep, FC, "right")
    ep, E, C, ta, te = ep[:lim], E[:lim], C[:lim], ta[:lim], te[:lim]
    im = fs(ta >= 0.99)
    if im < 0:
        return None, "no_mem"
    if fs(te >= 0.9) >= 0:
        return None, "violated"
    t_mem = int(ep[im])
    ifc = len(ep) - 1
    Efc = E[ifc]
    order = np.argsort(Efc)[::-1]
    comm = order[:4].tolist()
    junk = [j for j in range(E.shape[1]) if j not in comm]
    r = E[:, junk].sum(1) / E.sum(1)
    r_mem = float(r[im])
    w = np.where(ep >= t_mem)[0]
    if len(w) < 5:
        return None, "no_mem"
    x, y = ep[w].astype(float), np.log(np.clip(r[w], 1e-12, None))
    lam = -float(np.polyfit(x, y, 1)[0])
    quad = float(np.polyfit(x, y, 2)[0])
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
    harm = harmonic_set([c + 1 for c in comm])
    hmask = np.array([(j + 1) in harm for j in junk])
    coh = np.abs(C[ifc]) / np.sqrt(np.clip(Efc, 1e-12, None))
    i50 = fs(ta >= 0.5)
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
        test_loss_fc=np.nan, test_loss_slope=np.nan,
    )
    if test_losses is not None and len(test_losses) >= 6:
        f["test_loss_fc"] = float(np.log10(max(test_losses[5], 1e-12)))
        f["test_loss_slope"] = float(test_losses[5] - test_losses[2])
    return f, "ok"


def predict(model, f):
    """Locked-weight ridge on standardized features; missing values imputed
    with the training-set medians stored in the model file."""
    x = np.array([f[k] if np.isfinite(f.get(k, np.nan)) else model["medians"][k]
                  for k in model["features"]], float)
    z = (x - np.array(model["mean"])) / np.array(model["scale"])
    resid = float(np.dot(z, model["coef"]) + model["intercept"])
    log_t = f["log_t_law"] + resid
    return 10 ** log_t


def load_model(path):
    return json.loads(open(path).read())
