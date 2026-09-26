"""Float64 measurements of a trained or constructed network (NORMS A6: measurement is float64).

Every function takes ONE network: U (n, m, p) and W (m, p) (numpy or torch; cast to float64), or its
logit table T with shape (p,)*n + (p,) (T[a_1, ..., a_n, c] = f(x)[c]).

Paper references (see core/THEORY_SPEC.md): Fourier convention prelim.tex; clean amplitude
beta_j = 2 p^{-nu} fhat(j,..,j;-j) approach.tex eq:spectrum / appendix app:spectrum; junk modes
beta_{j,T} cyclic.tex; coverage and the phase lemma approach.tex lem:phase; ceilings approach.tex
thm:universal, cyclic.tex thm:cyclic, app:models.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import torch

from . import models as M
from . import programme as prog


# ------------------------------------------------------------------------------------------
# conversion helpers
# ------------------------------------------------------------------------------------------

def _np64(x) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().to("cpu", torch.float64).numpy()
    return np.asarray(x, dtype=np.float64)


def squeeze_net(U, W):
    """Accept (1, n, m, p)/(1, m, p) or (n, m, p)/(m, p); return float64 numpy (n, m, p), (m, p)."""
    U, W = _np64(U), _np64(W)
    if U.ndim == 4:
        assert U.shape[0] == 1
        U = U[0]
    if W.ndim == 3:
        assert W.shape[0] == 1
        W = W[0]
    return U, W


def table(model: str, U, W) -> np.ndarray:
    """Float64 logit table, shape (p,)*n + (p,)."""
    U, W = squeeze_net(U, W)
    n, m, p = U.shape
    Ut = torch.from_numpy(U)[None]
    Wt = torch.from_numpy(W)[None]
    with torch.no_grad():
        F = M.logits(model, Ut, Wt)[0].numpy()
    return F.reshape((p,) * (n + 1))


def label_table(p: int, n: int) -> np.ndarray:
    """y(x) on the grid, shape (p,)*n."""
    idx = np.indices((p,) * n).sum(0)
    return idx % p


# ------------------------------------------------------------------------------------------
# margins
# ------------------------------------------------------------------------------------------

def offset_gaps(T: np.ndarray) -> np.ndarray:
    """G[x..., delta] = f(x)[y] - f(x)[y - delta] for delta = 0..p-1 (delta = 0 column is 0)."""
    p = T.shape[-1]
    n = T.ndim - 1
    y = label_table(p, n)
    flat = T.reshape(-1, p)
    yy = y.reshape(-1)
    rows = np.arange(len(yy))
    fy = flat[rows, yy]
    G = np.empty((len(yy), p))
    for d in range(p):
        G[:, d] = fy - flat[rows, (yy - d) % p]
    return G.reshape(T.shape[:-1] + (p,))


def per_offset_margins(T: np.ndarray) -> np.ndarray:
    """h_delta = min_x f(x)[y(x)] - f(x)[y(x) - delta], for delta = 1..p-1 (array of length p-1)."""
    p = T.shape[-1]
    n = T.ndim - 1
    flat = T.reshape(-1, p)
    yy = label_table(p, n).reshape(-1)
    rows = np.arange(len(yy))
    fy = flat[rows, yy]
    out = np.empty(p - 1)
    for d in range(1, p):
        out[d - 1] = (fy - flat[rows, (yy - d) % p]).min()
    return out


def margin(T: np.ndarray) -> float:
    """h(theta) = min_x min_{c != y(x)} f(x)[y(x)] - f(x)[c] (full table)."""
    return float(per_offset_margins(T).min())


def per_offset_margins_folded(T: np.ndarray) -> np.ndarray:
    """min(h_delta, h_{p-delta}) for delta = 1..h (the folded offsets of the programme)."""
    hd = per_offset_margins(T)
    p = len(hd) + 1
    h = (p - 1) // 2
    return np.minimum(hd[:h], hd[::-1][:h])


def l2q_norm(U, W, q: float) -> float:
    """||theta||_{2,q} = (sum_i ||omega_i||^q)^{1/q}, float64."""
    U, W = squeeze_net(U, W)
    sq = (U ** 2).sum(axis=(0, 2)) + (W ** 2).sum(axis=1)
    return float((sq ** (q / 2)).sum() ** (1.0 / q))


def normalized_margin(T: np.ndarray, U, W, q: float) -> float:
    """gamma = h(theta) / ||theta||_{2,q}^nu with nu = n + 1 (scale invariant)."""
    nu = T.ndim
    return margin(T) / l2q_norm(U, W, q) ** nu


# ------------------------------------------------------------------------------------------
# ceilings
# ------------------------------------------------------------------------------------------

def B_p(p: int, nu: int) -> float:
    """B_p = 2 (2 p nu)^{-nu/2} (approach.tex, after the definition of C_b)."""
    return 2.0 * (2.0 * p * nu) ** (-nu / 2.0)


def ceilings(p: int, n: int, q: float, Gamma: float) -> dict:
    """The three ceilings for (p, n, q) given Gamma = Gamma_s(p), s = nu/q.

    universal_product = B_p Gamma                    (thm:universal, product networks)
    class_IF          = nu^{-(nu-q)/q} B_p Gamma     (thm:cyclic, interference-free product networks)
    universal_power   = n! B_p Gamma                 (thm:universal + app:models, power networks)
    """
    nu = n + 1
    B = B_p(p, nu)
    return {
        "universal_product": B * Gamma,
        "class_IF": nu ** (-(nu - q) / q) * B * Gamma,
        "universal_power": math.factorial(n) * B * Gamma,
    }


# ------------------------------------------------------------------------------------------
# spectrum
# ------------------------------------------------------------------------------------------

def sign_patterns(n: int):
    """All T subsets of [n] as sign vectors eps in {+1,-1}^n (eps_t = -1 iff t in T), T = {} first."""
    return [tuple(e) for e in itertools.product((1, -1), repeat=n)]


def spectrum(T: np.ndarray) -> dict:
    """Clean amplitudes, junk modes and energy split of a logit table (float64).

    Returns a dict with
      beta      (h,) complex: beta_j = 2 p^{-nu} fhat(j,..,j; -j)      (approach.tex eq:spectrum)
      A         (h,) float  : |beta_j|
      junk      dict {eps (tuple of +-1, not all +1): (h,) complex}:
                beta_{j,T} = 2 p^{-nu} fhat(eps_1 j, .., eps_n j; -j)   (cyclic.tex, T = {t: eps_t = -1})
      junk_abs  (h,) float  : sqrt(sum_T |beta_{j,T}|^2) (root-sum-square of the 2^n - 1 junk modes)
      energy    dict of table energies sum_{x,c} f^2 split by Parseval (each mode counted with its
                conjugate):
                  total     = sum f^2
                  clean     = sum_j 2 |fhat(j..j;-j)|^2 / p^nu
                  junk      = sum_j sum_{T != {}} 2 |fhat(eps j; -j)|^2 / p^nu
                  diag      = clean + junk (all modes (+-j, .., +-j; +-j), j in F)
                  zero_mode = |fhat(0,..,0)|^2 / p^nu (the constant)
                  off_diag  = total - diag - zero_mode (everything else: cross-frequency modes and
                              modes with some zero slot)
      frac      the same energies divided by total.
    """
    p = T.shape[-1]
    n = T.ndim - 1
    nu = n + 1
    h = (p - 1) // 2
    Fh = np.fft.fftn(T)
    js = np.arange(1, h + 1)
    idx_clean = tuple([js] * n + [(-js) % p])
    beta = 2.0 * p ** (-nu) * Fh[idx_clean]
    junk = {}
    E_junk = 0.0
    for eps in sign_patterns(n)[1:]:
        idx = tuple([(e * js) % p for e in eps] + [(-js) % p])
        g = Fh[idx]
        junk[eps] = 2.0 * p ** (-nu) * g
        E_junk += 2.0 * (np.abs(g) ** 2).sum() / p ** nu
    E_clean = 2.0 * (np.abs(Fh[idx_clean]) ** 2).sum() / p ** nu
    E_total = float((T ** 2).sum())
    E_zero = float(np.abs(Fh[(0,) * nu]) ** 2 / p ** nu)
    junk_abs = np.sqrt(sum(np.abs(v) ** 2 for v in junk.values())) if junk else np.zeros(h)
    E = {"total": E_total, "clean": float(E_clean), "junk": float(E_junk), "diag": float(E_clean + E_junk),
         "zero_mode": E_zero, "off_diag": float(E_total - E_clean - E_junk - E_zero)}
    frac = {k: (v / E_total if E_total > 0 else float("nan")) for k, v in E.items()}
    return {"beta": beta, "A": np.abs(beta), "junk": junk, "junk_abs": junk_abs, "energy": E, "frac": frac}


def neuron_fourier(U, W) -> dict:
    """Per-neuron Fourier content (float64).

    For every neuron i, slot t (operands 1..n, then the readout as slot n) and folded frequency j:
      mag[i, t, j] = |vhat(j)| of the slot's weight vector (j = 0..h, j = 0 is the constant)
      energy share of each frequency in the neuron: E[i, j] = sum_t |vhat_t(j)|^2 * (1 if j = 0
      else 2) / (p ||omega_i||^2)   (Parseval, so sum_j E[i, j] = 1 for a non-zero neuron)
      beta_n[i, j] = 2 p^{-nu} prod_t uhat_t(j) * conj(what(j)) (the product-neuron clean coefficient;
      multiply by n! for a power neuron).
    Also returns dominant[i] = argmax_j>=1 E[i, j] (1-based frequency) and purity[i] = E[i, dom] /
    sum_{j>=1} E[i, j].
    """
    U, W = squeeze_net(U, W)
    n, m, p = U.shape
    nu = n + 1
    h = (p - 1) // 2
    V = np.concatenate([U, W[None]], axis=0)  # (nu, m, p)
    Vh = np.fft.fft(V, axis=2)  # (nu, m, p)
    mag = np.abs(Vh[:, :, : h + 1]).transpose(1, 0, 2)  # (m, nu, h+1)
    sq = (V ** 2).sum(axis=(0, 2))  # (m,)
    E = (mag ** 2).sum(1)
    E[:, 1:] *= 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        E = E / (p * sq[:, None])
    js = np.arange(1, h + 1)
    prod = np.prod(Vh[:n][:, :, js], axis=0) * np.conj(Vh[n][:, js])  # (m, h)
    beta_n = 2.0 * p ** (-nu) * prod
    nz = E[:, 1:].sum(1)
    dom = np.argmax(E[:, 1:], axis=1) + 1
    with np.errstate(invalid="ignore", divide="ignore"):
        purity = E[np.arange(m), dom] / nz
    return {"mag": mag, "E": E, "beta_neuron": beta_n, "dominant": dom, "purity": purity,
            "sqnorm": sq}


# ------------------------------------------------------------------------------------------
# derived quantities
# ------------------------------------------------------------------------------------------

def budget_shares(A: np.ndarray, s: float) -> np.ndarray:
    """w_j = A_j^{1/s} / sum_k A_k^{1/s}."""
    A = np.clip(np.asarray(A, dtype=float), 0, None)
    a = A ** (1.0 / s)
    tot = a.sum()
    return a / tot if tot > 0 else a


def budget_normalize(A: np.ndarray, s: float) -> np.ndarray:
    """yhat = A / ||A||_{1/s} with ||A||_{1/s} = (sum A^{1/s})^s, so sum yhat^{1/s} = 1."""
    A = np.clip(np.asarray(A, dtype=float), 0, None)
    c = (A ** (1.0 / s)).sum() ** s
    return A / c if c > 0 else A


def clean_coverage_margin(A_or_beta: np.ndarray, p: int) -> float:
    """min_delta sum_j a_j cov_j(delta), delta = 1..h.

    With a = A = |beta| this is the right-hand side of the phase lemma (lem:phase); with a = Re(beta)
    it is the middle term. Both upper-bound the margin h(theta).
    """
    a = np.asarray(A_or_beta, dtype=float)
    return float((prog.coverage_matrix(p) @ a).min())


def ablate_frequency(U, W, j: int):
    """Remove frequency +-j from every weight vector (all operand slots and the readout).

    Precisely: for every vector v in R^p among U[t, i, :] and W[i, :], set vhat(j) = vhat(-j) = 0 and
    invert, i.e. v <- v - P_j v with P_j the orthogonal projection onto span{cos(2 pi j a / p),
    sin(2 pi j a / p)}. Returns new float64 arrays (U', W').
    """
    U, W = squeeze_net(U, W)
    p = U.shape[-1]
    a = np.arange(p)
    c = np.cos(2 * np.pi * j * a / p)
    s_ = np.sin(2 * np.pi * j * a / p)
    # c and s_ are orthogonal with squared norm p/2 each (p odd, j != 0 mod p)
    def proj(v):
        return v - np.multiply.outer(v @ c, c) * (2.0 / p) - np.multiply.outer(v @ s_, s_) * (2.0 / p)
    return proj(U), proj(W)


def ablation_margins(model: str, U, W) -> dict:
    """Margin after ablating each frequency j = 1..h (see ablate_frequency).

    Returns {'h_base', 'h_ablated' (h,), 'rel_loss' (h,) = 1 - h_ablated / h_base}. Raw margins
    (no renormalisation), which is the quantity bounded in CONSEQUENCES §3.
    """
    U, W = squeeze_net(U, W)
    p = U.shape[-1]
    h = (p - 1) // 2
    base = margin(table(model, U, W))
    out = np.empty(h)
    for j in range(1, h + 1):
        Ua, Wa = ablate_frequency(U, W, j)
        out[j - 1] = margin(table(model, Ua, Wa))
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = 1.0 - out / base
    return {"h_base": base, "h_ablated": out, "rel_loss": rel}


def counts(A: np.ndarray, s: float) -> dict:
    """Feature counts of an amplitude vector under every registered definition.

      thr_1, thr_5, thr_10, thr_20 : #{j : A_j >= t * max A} for t = 1%, 5%, 10%, 20%
      pr_amp    : (sum A)^2 / sum A^2                           (participation ratio of amplitudes)
      pr_budget : (sum A^{1/s})^2 / sum A^{2/s}                 (participation ratio of budget shares)
      mass_90, mass_95, mass_99 : smallest k such that the k largest budget shares w_j sum to >= 90/95/99%
    """
    A = np.clip(np.asarray(A, dtype=float), 0, None)
    out = {}
    mx = A.max() if A.size else 0.0
    for t in (1, 5, 10, 20):
        out[f"thr_{t}"] = int((A >= (t / 100.0) * mx).sum()) if mx > 0 else 0
    out["pr_amp"] = float(A.sum() ** 2 / (A ** 2).sum()) if mx > 0 else 0.0
    a = A ** (1.0 / s)
    out["pr_budget"] = float(a.sum() ** 2 / (a ** 2).sum()) if mx > 0 else 0.0
    w = np.sort(budget_shares(A, s))[::-1]
    cw = np.cumsum(w)
    for t in (90, 95, 99):
        out[f"mass_{t}"] = int(np.searchsorted(cw, t / 100.0 - 1e-12) + 1) if mx > 0 else 0
    return out


COUNT_KEYS = ("thr_1", "thr_5", "thr_10", "thr_20", "pr_amp", "pr_budget", "mass_90", "mass_95", "mass_99")


def full_measurement(model: str, U, W, q: float, with_ablation: bool = True, Gamma: float | None = None) -> dict:
    """Every registered measurement of one network, as a JSON-friendly dict (float64)."""
    U, W = squeeze_net(U, W)
    n, m, p = U.shape
    nu = n + 1
    s = nu / q
    T = table(model, U, W)
    h_ = margin(T)
    nrm = l2q_norm(U, W, q)
    gam = h_ / nrm ** nu
    sp = spectrum(T)
    nf = neuron_fourier(U, W)
    out = {
        "model": model, "n": n, "nu": nu, "p": p, "q": q, "s": s, "width": m,
        "margin": h_, "l2q_norm": nrm, "gamma": gam,
        "accuracy": float((T.reshape(-1, p).argmax(1) == label_table(p, n).reshape(-1)).mean()),
        "beta_re": sp["beta"].real.tolist(), "beta_im": sp["beta"].imag.tolist(),
        "A": sp["A"].tolist(),
        "junk_abs": sp["junk_abs"].tolist(),
        "junk": {"".join("+" if e > 0 else "-" for e in k): np.abs(v).tolist() for k, v in sp["junk"].items()},
        "energy": sp["energy"], "energy_frac": sp["frac"],
        "budget_shares": budget_shares(sp["A"], s).tolist(),
        "yhat": budget_normalize(sp["A"], s).tolist(),
        "clean_cov_margin_abs": clean_coverage_margin(sp["A"], p),
        "clean_cov_margin_re": clean_coverage_margin(sp["beta"].real, p),
        "per_offset_margin": per_offset_margins(T).tolist(),
        "per_offset_margin_folded": per_offset_margins_folded(T).tolist(),
        "counts": counts(sp["A"], s),
        "neuron_dominant": nf["dominant"].tolist(),
        "neuron_purity": np.nan_to_num(nf["purity"]).tolist(),
        "neuron_sqnorm": nf["sqnorm"].tolist(),
        "neuron_active_per_freq": np.bincount(nf["dominant"][nf["sqnorm"] > 1e-12 * max(nf["sqnorm"].max(), 1e-300)],
                                              minlength=(p - 1) // 2 + 1)[1:].tolist(),
    }
    if Gamma is not None:
        ce = ceilings(p, n, q, Gamma)
        out["Gamma"] = Gamma
        out["ceilings"] = ce
        out["ratio_to_ceiling"] = {k: gam / v for k, v in ce.items()}
    if with_ablation:
        ab = ablation_margins(model, U, W)
        out["ablation"] = {"h_base": ab["h_base"], "h_ablated": ab["h_ablated"].tolist(),
                           "rel_loss": ab["rel_loss"].tolist()}
    return out
