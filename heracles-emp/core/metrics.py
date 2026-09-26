"""Agreement metrics between a measured network and the programme's prediction (NORMS E16).

Every prediction is scored with ALL of the metrics below, never a hand-picked one. All per-run metrics
are computed AFTER the best alignment of the prediction over dilations and over all optimal vertices:

    alignment rule (declared): among the rows y* of `y_all` (every optimal vertex of the programme and
    all of its dilates j -> fold(u j)), choose the one maximising the cosine similarity between the
    measured amplitude vector A and y*; ties are broken by the smaller total-variation distance
    between budget shares.

Notation: A_j = |beta_j| (measured clean amplitudes, j = 1..h), s = nu/q,
    yhat = A / ||A||_{1/s}   (||A||_{1/s} = (sum A^{1/s})^s, so sum yhat^{1/s} = 1: the measured
                              amplitudes put on the programme's budget)
    w    = A^{1/s} / sum A^{1/s}  (measured budget shares),   w* = y*^{1/s} (predicted, sums to 1).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

from . import measure as MS
from . import programme as prog

THRESHOLDS = (1, 5, 10, 20)


# ----------------------------------------------------------------------------------------------
# elementary statistics (each is a pure function of two equal-length vectors)
# ----------------------------------------------------------------------------------------------

def r2_identity(meas, pred) -> float:
    """R^2 against the identity line: 1 - sum (meas - pred)^2 / sum (meas - mean(meas))^2.

    'How much of the variance of the measured values is explained by predicting them exactly by
    pred' (no fitted slope or intercept). Can be negative. nan if meas is constant.
    """
    meas, pred = np.asarray(meas, float), np.asarray(pred, float)
    ss_tot = ((meas - meas.mean()) ** 2).sum()
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - ((meas - pred) ** 2).sum() / ss_tot)


def r2_linear_fit(meas, pred) -> float:
    """R^2 of the fitted OLS regression meas ~ a + b * pred (= squared Pearson correlation).

    Scale- and offset-free: it can be high even when meas != pred. nan if either is constant.
    """
    r = pearson(meas, pred)
    return float(r * r) if np.isfinite(r) else float("nan")


def pearson(a, b) -> float:
    """Pearson correlation; nan if either vector is constant."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b) -> float:
    """Spearman rank correlation (average ranks for ties); nan if either vector is constant."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)


def kendall(a, b) -> float:
    """Kendall tau-b; nan if either vector is constant."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(stats.kendalltau(a, b).statistic)


def cosine(a, b) -> float:
    """Cosine similarity a.b / (|a| |b|); nan if either is zero."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(a @ b / (na * nb))


def tv_distance(w1, w2) -> float:
    """Total-variation distance 0.5 * sum |w1 - w2| between two probability vectors."""
    return float(0.5 * np.abs(np.asarray(w1, float) - np.asarray(w2, float)).sum())


def max_abs_error(meas, pred) -> float:
    """max_j |meas_j - pred_j|."""
    return float(np.abs(np.asarray(meas, float) - np.asarray(pred, float)).max())


def jaccard(s1, s2) -> float:
    """|s1 & s2| / |s1 | s2| of two sets (1.0 if both are empty)."""
    s1, s2 = set(s1), set(s2)
    u = s1 | s2
    return 1.0 if not u else len(s1 & s2) / len(u)


def linfit_ci(x, y, alpha: float = 0.05) -> dict:
    """OLS y = intercept + slope * x with (1 - alpha) t-based confidence intervals.

    Returns slope, intercept, their standard errors and CIs, r (Pearson), n. nan-filled if n < 3 or x
    is constant.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    out = {"n": n}
    if n < 3 or x.std() == 0:
        return {**out, "slope": float("nan"), "intercept": float("nan"), "slope_ci": [float("nan")] * 2,
                "intercept_ci": [float("nan")] * 2, "r": float("nan")}
    res = stats.linregress(x, y)
    tq = stats.t.ppf(1 - alpha / 2, n - 2)
    return {**out, "slope": float(res.slope), "intercept": float(res.intercept),
            "slope_se": float(res.stderr), "intercept_se": float(res.intercept_stderr),
            "slope_ci": [float(res.slope - tq * res.stderr), float(res.slope + tq * res.stderr)],
            "intercept_ci": [float(res.intercept - tq * res.intercept_stderr),
                             float(res.intercept + tq * res.intercept_stderr)],
            "r": float(res.rvalue)}


def frac_within(meas, pred, tol: float = 1.0) -> float:
    """Fraction of items with |meas - pred| <= tol (default: within +-1, for counts)."""
    meas, pred = np.asarray(meas, float), np.asarray(pred, float)
    return float((np.abs(meas - pred) <= tol + 1e-12).mean()) if len(meas) else float("nan")


def bootstrap_over_seeds(values_by_seed, stat=np.mean, n_boot: int = 10000, alpha: float = 0.05,
                         rng_seed: int = 0) -> dict:
    """Percentile bootstrap over seeds: resample the seed list with replacement.

    values_by_seed: list (one entry per seed) of scalars or of arrays (e.g. all runs of that seed);
    stat maps the concatenated resample to a scalar. Returns {'stat', 'ci', 'n_seeds', 'n_boot'}.
    """
    vals = [np.atleast_1d(np.asarray(v, float)) for v in values_by_seed]
    k = len(vals)
    if k == 0:
        return {"stat": float("nan"), "ci": [float("nan")] * 2, "n_seeds": 0, "n_boot": n_boot}
    rng = np.random.default_rng(rng_seed)
    point = float(stat(np.concatenate(vals)))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, k, k)
        boots[b] = stat(np.concatenate([vals[i] for i in idx]))
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"stat": point, "ci": [float(lo), float(hi)], "n_seeds": k, "n_boot": n_boot}


def sweep_summary(meas, pred, tol: float = 1.0) -> dict:
    """Sweep-level agreement of paired scalars (e.g. measured vs predicted K over cells):
    both R^2 kinds, Pearson, Spearman, Kendall, OLS slope/intercept with 95% CIs, fraction within
    +-tol, exact fraction, mean signed and absolute error."""
    meas, pred = np.asarray(meas, float), np.asarray(pred, float)
    return {
        "n": int(len(meas)),
        "r2_identity": r2_identity(meas, pred),
        "r2_linear_fit": r2_linear_fit(meas, pred),
        "pearson": pearson(meas, pred),
        "spearman": spearman(meas, pred),
        "kendall": kendall(meas, pred),
        "linfit_meas_on_pred": linfit_ci(pred, meas),
        "frac_within_tol": frac_within(meas, pred, tol),
        "tol": tol,
        "frac_exact": frac_within(meas, pred, 0.0),
        "mean_signed_error": float((meas - pred).mean()) if len(meas) else float("nan"),
        "mean_abs_error": float(np.abs(meas - pred).mean()) if len(meas) else float("nan"),
    }


# ----------------------------------------------------------------------------------------------
# per-run scoring
# ----------------------------------------------------------------------------------------------

def align(A, y_all, s: float) -> dict:
    """Best alignment of the measured amplitudes A to a row of y_all (declared rule, module docstring)."""
    A = np.asarray(A, float)
    Y = np.atleast_2d(np.asarray(y_all, float))
    w = MS.budget_shares(A, s)
    best = None
    for i, yv in enumerate(Y):
        c = cosine(A, yv)
        tv = tv_distance(w, MS.budget_shares(yv, s))
        key = (-(c if np.isfinite(c) else -np.inf), tv)
        if best is None or key < best[0]:
            best = (key, i)
    return {"index": int(best[1]), "cosine": -best[0][0], "tv": best[0][1]}


def score_amplitudes(A, s: float, y_all, kstar=None) -> dict:
    """Every registered agreement metric between measured amplitudes A and the prediction y_all.

    Returns a flat-ish dict:
      alignment             : chosen row, its cosine and TV
      counts[def]           : {'meas', 'pred_same_def', 'kstar', 'signed', 'abs', 'exact', 'signed_vs_kstar'}
                              for every definition in measure.COUNT_KEYS ('pred_same_def' applies the
                              same definition to y*; 'exact' is meas == pred_same_def for integer
                              definitions and |diff| <= 1e-6 for the participation ratios, because
                              float64 noise amplitudes ~1e-17 enter the budget as (1e-17)^{1/s} ~ 1e-8)
      amp.*                 : on yhat vs y* (budget-normalised amplitudes): r2_identity, r2_linear_fit,
                              pearson, spearman, cosine, max_abs_error, l1_error, slope/intercept
      share.*               : on w vs w*: tv, r2_identity, r2_linear_fit, pearson, spearman, cosine,
                              max_abs_error
      jaccard_thr_t         : Jaccard of {j: A_j >= t% max A} and spt(y*), t in 1, 5, 10, 20
    """
    A = np.clip(np.asarray(A, float), 0, None)
    Y = np.atleast_2d(np.asarray(y_all, float))
    al = align(A, Y, s)
    ys = Y[al["index"]]
    K = int((ys > 0).sum()) if kstar is None else int(kstar)
    yhat = MS.budget_normalize(A, s)
    w = MS.budget_shares(A, s)
    ws = MS.budget_shares(ys, s)
    cm, cp = MS.counts(A, s), MS.counts(ys, s)
    counts = {}
    for k in MS.COUNT_KEYS:
        d = cm[k] - cp[k]
        tol = 0 if isinstance(cm[k], int) else 1e-6  # float definitions: float64 noise floor (see docstring)
        counts[k] = {"meas": cm[k], "pred_same_def": cp[k], "kstar": K, "signed": d, "abs": abs(d),
                     "exact": bool(abs(d) <= tol), "signed_vs_kstar": cm[k] - K}
    lf = linfit_ci(ys, yhat)
    out = {
        "alignment": al,
        "y_star": ys.tolist(),
        "kstar": K,
        "counts": counts,
        "amp": {"r2_identity": r2_identity(yhat, ys), "r2_linear_fit": r2_linear_fit(yhat, ys),
                "pearson": pearson(yhat, ys), "spearman": spearman(yhat, ys), "cosine": cosine(yhat, ys),
                "max_abs_error": max_abs_error(yhat, ys), "l1_error": float(np.abs(yhat - ys).sum()),
                "slope": lf["slope"], "intercept": lf["intercept"]},
        "share": {"tv": tv_distance(w, ws), "r2_identity": r2_identity(w, ws), "r2_linear_fit": r2_linear_fit(w, ws),
                  "pearson": pearson(w, ws), "spearman": spearman(w, ws), "cosine": cosine(w, ws),
                  "max_abs_error": max_abs_error(w, ws)},
    }
    mx = A.max() if A.size else 0.0
    spt = set(np.flatnonzero(ys > 0).tolist())
    for t in THRESHOLDS:
        out[f"jaccard_thr_{t}"] = jaccard(set(np.flatnonzero(A >= t / 100.0 * mx).tolist()) if mx > 0 else set(), spt)
    return out


def prediction(p: int, s: float) -> dict:
    """The programme's prediction at (p, s) with its certification label (core.programme.solve_any)."""
    return prog.solve_any(p, s)


def score_run(measurement: dict, pred: dict | None = None) -> dict:
    """Score one measured network (core.measure.full_measurement output) against the programme.

    Adds the margin ratios gamma / ceiling for every ceiling. The prediction's certification label is
    carried along ('exact (vertex cache)', 'exact (certified K cap)' or 'best-found (not certified)').
    """
    p, s, q, n = measurement["p"], measurement["s"], measurement["q"], measurement["n"]
    if pred is None:
        pred = prediction(p, s)
    sc = score_amplitudes(measurement["A"], s, pred["y_all"], None)
    ce = MS.ceilings(p, n, q, pred["gamma"])
    sc["Gamma"] = pred["gamma"]
    sc["prediction_label"] = pred["label"]
    sc["kstar_all"] = pred.get("K_all")
    sc["ceilings"] = ce
    sc["gamma_over_ceiling"] = {k: measurement["gamma"] / v for k, v in ce.items()}
    return sc
