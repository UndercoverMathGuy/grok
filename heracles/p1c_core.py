"""P1C / shared core: the (KL) atom system, its minimisation, and the stationarity data.

Symbols (all defined again in P1C.md):
  n      number of operand slots; nu = n+1 is the homogeneity degree
  N      number of atoms (= neurons carrying one frequency, or their Fourier data)
  v[l,j] unit-modulus phase of atom l in slot j  (stored as an angle phi[l,j])
  c[l]   complex "clean amplitude" of atom l
  A      the clean total (we always normalise A = 1)
  r      = q/nu in (0,1]; the cost is  F = sum_l |c_l|^r
  (KL)   sum_l c_l prod_{j in T} v_lj = A [T empty], for every subset T of {1..n}

  Lambda(z) = sum_T lam_T prod_{j in T} z_j    a multi-affine "test polynomial";
             NORMALISED means lam_empty = 1 (its torus average is 1).

Two facts this module implements and checks.

(1) STATIONARITY.  At a local minimum of F subject to (KL), with the zero atoms deleted,
    there is a normalised multi-affine Lambda with

      (i)   c_l * Lambda(v_l) = |c_l|^r / F        (so c_l Lambda(v_l) > 0 and
                                                    |Lambda(v_l)| = |c_l|^{r-1} / F)
      (ii)  for every atom l and slot j, writing Lambda(z) = alpha_lj + z_j beta_lj
            with alpha, beta evaluated at the other slots of v_l,
            alpha_lj / Lambda(v_l)  is REAL   (equivalently v_lj beta_lj / Lambda(v_l) is real):
            each atom sits at a critical point of |Lambda| along every slot circle.

    Plain reading of (i): the dual test polynomial is small exactly where the atoms are big.

(2) THE VALUE IDENTITY.  (i) alone forces, with sigma := r/(1-r),

      F  =  ( sum_l |Lambda(v_l)|^{-sigma} )^{1-r} ,

    so P1 (at stationary points) is EXACTLY the statement
    sum_l |Lambda(v_l)|^{-sigma} >= n+1.

Local CPU, float64.
"""
import itertools
import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------- basic algebra


def subsets(n):
    """All subsets T of {0..n-1} as tuples, ordered by the integer mask."""
    return [tuple(j for j in range(n) if (m >> j) & 1) for m in range(1 << n)]


def Xmat(phi):
    """X[l, m] = prod_{j in T_m} v_lj, with T_m the subset whose mask is m.
    phi: (N, n) real angles.  Returns (N, 2^n) complex."""
    N, n = phi.shape
    v = np.exp(1j * phi)
    X = np.ones((N, 1 << n), dtype=complex)
    for m in range(1 << n):
        for j in range(n):
            if (m >> j) & 1:
                X[:, m] *= v[:, j]
    return X


def kl_residual(phi, c, A=1.0):
    """The 2^n complex numbers C_T - A[T empty]."""
    X = Xmat(phi)
    n = phi.shape[1]
    b = np.zeros(1 << n, dtype=complex)
    b[0] = A
    return X.T @ c - b


def cost(c, r, eps=0.0):
    return float(np.sum((np.abs(c) ** 2 + eps) ** (r / 2.0)))


def eval_multiaffine(lam, phi):
    """Lambda(v_l) for every atom l.  lam: (2^n,) complex, phi: (N, n)."""
    return Xmat(phi) @ lam


# ---------------------------------------------------------------- minimisation


def _pack(phi, c):
    return np.concatenate([phi.ravel(), c.real, c.imag])


def _unpack(x, N, n):
    phi = x[: N * n].reshape(N, n)
    c = x[N * n: N * n + N] + 1j * x[N * n + N:]
    return phi, c


def minimise_F(N, n, r, seed=0, A=1.0, eps_schedule=(1e-3, 1e-6, 1e-9, 1e-12, 0.0),
               x0=None, maxiter=400):
    """SLSQP minimisation of sum_l (|c_l|^2 + eps)^{r/2} subject to (KL), annealing eps -> 0."""
    rng = np.random.default_rng(seed)
    if x0 is None:
        phi = rng.uniform(0, 2 * np.pi, (N, n))
        # start from a least-squares feasible c for those phases
        X = Xmat(phi)
        b = np.zeros(1 << n, dtype=complex)
        b[0] = A
        c, *_ = np.linalg.lstsq(X.T, b, rcond=None)
        x0 = _pack(phi, c)

    def f(x, eps):
        _, c = _unpack(x, N, n)
        return float(np.sum((np.abs(c) ** 2 + eps) ** (r / 2.0)))

    def g(x, eps):
        phi, c = _unpack(x, N, n)
        gr = np.zeros_like(x)
        w = r * (np.abs(c) ** 2 + eps) ** (r / 2.0 - 1.0)
        gr[N * n: N * n + N] = w * c.real
        gr[N * n + N:] = w * c.imag
        return gr

    def con(x):
        phi, c = _unpack(x, N, n)
        z = kl_residual(phi, c, A)
        return np.concatenate([z.real, z.imag])

    x = x0.copy()
    for eps in eps_schedule:
        res = minimize(f, x, args=(eps,), jac=g, method="SLSQP",
                       constraints=[dict(type="eq", fun=con)],
                       options=dict(maxiter=maxiter, ftol=1e-14))
        x = res.x
    phi, c = _unpack(x, N, n)
    return phi, c, cost(c, r), float(np.abs(kl_residual(phi, c, A)).max())


# ---------------------------------------------------------------- stationarity


def dual_lambda(phi, c, r, tol=1e-7):
    """Solve condition (i) for the multi-affine coefficients lam (least squares on the
    support).  Returns (lam, support_mask, F)."""
    F = cost(c, r)
    sup = np.abs(c) > tol
    X = Xmat(phi)[sup]                      # (s, 2^n)
    rhs = (np.abs(c[sup]) ** r) / (F * c[sup])   # Lambda(v_l) required by (i)
    lam, *_ = np.linalg.lstsq(X, rhs, rcond=None)
    return lam, sup, F


def stationarity_report(phi, c, r, tol=1e-7):
    """All the diagnostics of (i), (ii) and the value identity."""
    n = phi.shape[1]
    lam, sup, F = dual_lambda(phi, c, r, tol)
    Lv = eval_multiaffine(lam, phi)
    out = {}
    out["F"] = F
    out["support"] = int(sup.sum())
    out["lam_empty"] = complex(lam[0])
    # (i) residual
    want = np.zeros_like(c)
    want[sup] = (np.abs(c[sup]) ** r) / (F * c[sup])
    out["res_i"] = float(np.abs(Lv[sup] - want[sup]).max()) if sup.any() else 0.0
    # (ii) residual: alpha_lj / Lambda(v_l) real
    v = np.exp(1j * phi)
    res_ii = 0.0
    for j in range(n):
        # alpha_lj = Lambda with z_j set to 0
        lam_a = lam.copy()
        for m in range(1 << n):
            if (m >> j) & 1:
                lam_a[m] = 0.0
        alpha = Xmat(phi) @ lam_a
        ratio = alpha[sup] / Lv[sup]
        res_ii = max(res_ii, float(np.abs(ratio.imag).max()))
    out["res_ii"] = res_ii
    # value identity
    if r < 1.0:
        sigma = r / (1.0 - r)
        S = float(np.sum(np.abs(Lv[sup]) ** (-sigma)))
        out["sum_Lambda_pow"] = S
        out["F_from_identity"] = S ** (1.0 - r)
    out["absL"] = np.abs(Lv).tolist()
    out["moduli"] = np.abs(c).tolist()
    return out
