"""p3b_max.py -- what the MAXIMISER actually looks like: junk, active set, KKT dual measure, gauge orbit.

Three measurements on the unrestricted whole-network max-margin solutions of `gopt_full.py`
(raw tensors E (n,m,p), W (p,m); no purity, no junk-free, no support restriction):

 (M1) same-|k| junk at the maximiser.  Atoms  c_ik = prod_j ehat_ij(k) * conj(what_i(k)),
      phases  v_ijk = conj(ehat_ij(k))/ehat_ij(k),  clean  G_k = sum_i c_ik,
      junk    J_{k,T} = sum_i c_ik prod_{j in T} v_ijk  (T a non-empty subset of the operand slots).
      Reported as the relative junk  eta_k = sqrt(sum_{T != 0} |J_{k,T}|^2) / |G_k|.
      Hypothesis (H) of Theorem D is  eta_k = 0 for every used frequency k.

 (M2) the active set and the KKT dual measure.  Phi(a,c) = z(a,s(a)) - z(a,c);  active pairs are
      those with Phi = margin.  A dual measure is any mu >= 0 on the active set with sum mu = 1 and
      sum_{(a,c)} mu(a,c) grad Phi(a,c) = lambda grad P_q,  lambda = nu*margin/(q*P_q);  found by
      non-negative least squares.  Reported: residual, and how far mu's a-marginal is from uniform.
      "mu uniform in a" is the premise of the dual-symmetrisation route; see P3B.md sec. 4.

 (M3) the per-neuron GAUGE torus.  R_phi^{(i)} rotates every positive-frequency Fourier coefficient
      of neuron i's n embeddings by phi and of its readout by n*phi.  Claim (Lemma G of P3B.md):
      it is orthogonal on every vector, so P_q is EXACTLY invariant; c_ik is invariant, hence every
      G_k is invariant; and J_{k,T} -> e^{-2i|T|phi} J_{k,T}.  Checked to fp64 here.

fp64, local CPU, no pods.
"""
import argparse, itertools, json, sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import gopt_full as GF  # noqa: E402
from kstar import ladder  # noqa: E402


# --------------------------------------------------------------------------------- atoms / junk
def atoms(E, W, p, n):
    """ehat (n,m,p), what (p,m) -> c (m, nf), v (n, m, nf), G (nf,), J dict[T] -> (nf,)"""
    nf = (p - 1) // 2
    Eh = np.fft.fft(E, axis=2)[:, :, 1:nf + 1]             # (n, m, nf), convention ehat(k)=sum_a e(a) w^{-ka}
    Wh = np.fft.fft(W, axis=0)[1:nf + 1, :].T              # (m, nf)
    c = np.prod(Eh, axis=0) * np.conj(Wh)                  # (m, nf)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(np.abs(Eh) > 1e-300, np.conj(Eh) / Eh, 1.0)
    G = c.sum(0)
    J = {}
    for rr in range(1, n + 1):
        for T in itertools.combinations(range(n), rr):
            J[T] = (c * np.prod(v[list(T)], axis=0)).sum(0)
    return c, v, G, J


def eta_of(G, J, tol=1e-12):
    Jm2 = sum(np.abs(j) ** 2 for j in J.values())
    return np.sqrt(Jm2) / np.maximum(np.abs(G), tol), np.abs(G)


# --------------------------------------------------------------------------------- gauge rotation
def gauge(E, W, p, n, phi, which=None):
    """R_phi^{(i)}: rotate positive-frequency coefficients of e_ij by phi, of w_i by n*phi."""
    nf = (p - 1) // 2
    m = W.shape[1]
    sel = np.ones(m, bool) if which is None else np.zeros(m, bool)
    if which is not None:
        sel[which] = True
    ph = np.ones(p, complex)
    ph[1:nf + 1] = np.exp(1j * phi)
    ph[p - nf:] = np.exp(-1j * phi)
    phn = np.ones(p, complex)
    phn[1:nf + 1] = np.exp(1j * n * phi)
    phn[p - nf:] = np.exp(-1j * n * phi)
    E2 = E.copy(); W2 = W.copy()
    Eh = np.fft.fft(E, axis=2); Wh = np.fft.fft(W, axis=0)
    Eh[:, sel, :] = Eh[:, sel, :] * ph[None, None, :]
    Wh[:, sel] = Wh[:, sel] * phn[:, None]
    E2 = np.real(np.fft.ifft(Eh, axis=2)); W2 = np.real(np.fft.ifft(Wh, axis=0))
    return E2, W2


# --------------------------------------------------------------------------------- network helpers
def logits(E, W, idx, n):
    H = np.ones((idx.shape[0], W.shape[1]))
    for j in range(n):
        H = H * E[j][:, idx[:, j]].T
    return H @ W.T, H


def margin_of(E, W, idx, y, n):
    z, _ = logits(E, W, idx, n)
    zz = z.copy(); zz[np.arange(len(y)), y] = -np.inf
    return float((z[np.arange(len(y)), y][:, None] - zz).min()), z


def clean_g(E, W, p, n, idx, y):
    """g(D) = mean over rows of z(a, s(a)-D); returns gamma_clean(D) = g(0)-g(D) for D=1..p-1."""
    z, _ = logits(E, W, idx, n)
    g = np.array([z[np.arange(len(y)), (y - D) % p].mean() for D in range(p)])
    return g[0] - g


# --------------------------------------------------------------------------------- KKT dual measure
def dual_measure(E, W, p, n, q, idx, y, tol):
    nu = n + 1
    m = W.shape[1]
    z, H = logits(E, W, idx, n)
    R = idx.shape[0]
    Phi = z[np.arange(R), y][:, None] - z                  # (R, p)
    Phi[np.arange(R), y] = np.inf
    mg = float(Phi.min())
    act = np.argwhere(Phi <= mg + tol * max(abs(mg), 1e-30))
    nrm2 = (E ** 2).sum((0, 2)) + (W ** 2).sum(0)
    Pq = float((nrm2 ** (q / 2)).sum())
    d = (q / 2) * nrm2 ** (q / 2 - 1)
    gP = np.concatenate([(2 * E * d[None, :, None]).reshape(-1), (2 * W * d[None, :]).reshape(-1)])
    lam = nu * mg / (q * Pq)
    # grad Phi(a,c) for each active pair
    A = np.zeros((len(act), gP.size))
    for t, (a, c) in enumerate(act):
        gE = np.zeros((n, m, p)); gW = np.zeros((p, m))
        gW[y[a]] += H[a]; gW[c] -= H[a]
        for j in range(n):
            Q = np.ones(m)
            for j2 in range(n):
                if j2 != j:
                    Q = Q * E[j2][:, idx[a, j2]]
            gE[j, :, idx[a, j]] += Q * (W[y[a]] - W[c])
        A[t] = np.concatenate([gE.reshape(-1), gW.reshape(-1)])
    # nnls for  A^T mu = lam gP  with sum mu = 1 (appended as a row, weighted)
    wsum = np.linalg.norm(lam * gP) + 1e-30
    M = np.vstack([A.T, wsum * np.ones((1, len(act)))])
    b = np.concatenate([lam * gP, [wsum]])
    mu, res = nnls(M, b)
    rel = res / (np.linalg.norm(b) + 1e-300)
    # a-marginal
    ma = np.zeros(R)
    for t, (a, c) in enumerate(act):
        ma[a] += mu[t]
    # is the UNIFORM measure on the active set a valid dual?  (premise of the symmetrisation route)
    muu = np.full(len(act), 1.0 / len(act))
    relu = np.linalg.norm(A.T @ muu - lam * gP) / (np.linalg.norm(lam * gP) + 1e-300)
    return mg, act, mu, rel, ma, Pq, float(relu)


# --------------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps", type=int, nargs="+", default=[5, 7])
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--qs", type=float, nargs="+", default=[2.0])
    ap.add_argument("--ms", type=int, nargs="+", default=[9])
    ap.add_argument("--restarts", type=int, default=20)
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--acttol", type=float, default=1e-6)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    n, nu = a.n, a.n + 1
    out = []
    for p in a.ps:
        for q in a.qs:
            for m in a.ms:
                idx, y = GF.rows_and_wrong(p, n)
                Bc = 2 * (2 * p * nu) ** (-nu / 2)
                Gs = GF.gamma_star(p, nu / q, a.kmax)
                tgt = nu ** (-(nu - q) / q) * Bc * Gs
                E, W = rerun_keep(p, n, q, m, a.restarts, a.seed, a.kmax, idx, y, tgt)
                mg, z = margin_of(E, W, idx, y, n)
                nrm2 = (E ** 2).sum((0, 2)) + (W ** 2).sum(0)
                Pq = float((nrm2 ** (q / 2)).sum())
                gq = mg / Pq ** (nu / q); ratio = gq / tgt
                c, v, G, J = atoms(E, W, p, n)
                eta, absG = eta_of(G, J)
                used = absG > 1e-3 * absG.max()
                gc = clean_g(E, W, p, n, idx, y)
                ar = np.arange(len(y))
                dev = np.array([gc[D] - (z[ar, y] - z[ar, (y - D) % p]).min() for D in range(1, p)])
                mg2, act, mu, rel, ma, _, relu = dual_measure(E, W, p, n, q, idx, y, a.acttol)
                R = idx.shape[0]
                unif = np.full(R, 1.0 / R)
                tv = 0.5 * np.abs(ma - unif).sum()
                row = dict(p=p, n=n, q=q, m=m, ratio=ratio, gamma_q=gq, target=tgt,
                           used_freqs=[int(k + 1) for k in np.where(used)[0]],
                           absG=absG[used].tolist(), eta=eta[used].tolist(),
                           max_eta_used=float(eta[used].max()) if used.any() else 0.0,
                           n_active=int(len(act)), n_rows=R,
                           frac_rows_active=float(len(set(act[:, 0].tolist())) / R),
                           kkt_rel_residual=float(rel),
                           amarg_TV_from_uniform=float(tv), uniform_dual_rel_residual=float(relu),
                           min_dev_over_D=float(dev.min()), max_dev_over_D=float(dev.max()),
                           clean_margin_min=float(gc[1:].min()), margin=mg)
                out.append(row)
                print(f"p={p} q={q:g} m={m}: gamma_q/target = {ratio:.12f}  freqs={row['used_freqs']}  "
                      f"max eta_k = {row['max_eta_used']:.3e}", flush=True)
                print(f"        active pairs {len(act)}/{R*(p-1)}, distinct rows "
                      f"{row['frac_rows_active']*100:.1f}%;  KKT nnls rel. residual {rel:.2e};  "
                      f"|a-marginal - uniform|_TV = {tv:.3e};  "
                      f"UNIFORM measure on the active set: KKT rel. residual {relu:.2e}", flush=True)
                print(f"        margin {mg:.6e}   min_D clean margin {gc[1:].min():.6e}   "
                      f"deficit range [{dev.min():.2e}, {dev.max():.2e}]", flush=True)
                # (M3) gauge check
                g1 = gauge_check(E, W, p, n, q)
                print(f"        [M3] gauge: dP_q={g1[0]:.2e}  dG={g1[1]:.2e}  "
                       f"junk phase law residual={g1[2]:.2e}", flush=True)
                row["gauge"] = dict(dPq=g1[0], dG=g1[1], junk_phase_resid=g1[2])
    d = ROOT / "FINAL_RUNS" / "P3B"; d.mkdir(parents=True, exist_ok=True)
    (d / f"max{a.tag}.json").write_text(json.dumps(out, indent=1, default=float))
    print("wrote", d / f"max{a.tag}.json")


def gauge_check(E, W, p, n, q, phi=0.7231, which=1):
    c0, v0, G0, J0 = atoms(E, W, p, n)
    nrm0 = (((E ** 2).sum((0, 2)) + (W ** 2).sum(0)) ** (q / 2)).sum()
    E2, W2 = gauge(E, W, p, n, phi, which=which)
    c1, v1, G1, J1 = atoms(E2, W2, p, n)
    nrm1 = (((E2 ** 2).sum((0, 2)) + (W2 ** 2).sum(0)) ** (q / 2)).sum()
    dP = abs(nrm1 - nrm0) / abs(nrm0)
    dG = float(np.abs(G1 - G0).max() / max(np.abs(G0).max(), 1e-300))
    # per-neuron law: neuron `which`'s contribution picks up exp(-2i|T|phi)
    resid = 0.0
    for T, j0 in J0.items():
        contrib0 = (c0[which] * np.prod(v0[list(T)][:, which], axis=0))
        contrib1 = (c1[which] * np.prod(v1[list(T)][:, which], axis=0))
        resid = max(resid, float(np.abs(contrib1 - np.exp(-2j * len(T) * phi) * contrib0).max()
                                 / max(np.abs(contrib0).max(), 1e-300)))
    return float(dP), dG, resid


def rerun_keep(p, n, q, m, restarts, seed, kmax, idx, y, tgt):
    """same search as gopt_full.run but returns the best (E, W)."""
    from scipy.optimize import minimize
    nu = n + 1
    rng = np.random.default_rng(seed)
    lad = ladder(p, min(kmax, (p - 1) // 2))
    starts = []
    for K in range(1, min(kmax, (p - 1) // 2, m // (n + 1)) + 1):
        S = lad["S"][K] if K > 1 else [1]
        E0, W0 = GF.build_block(p, n, S, m)
        for nz in (0.0, 0.05, 0.3):
            starts.append((E0 + nz * rng.normal(size=E0.shape), W0 + nz * rng.normal(size=W0.shape)))
    for _ in range(restarts):
        starts.append((rng.normal(size=(n, m, p)), rng.normal(size=(p, m))))
    best = (-1e18, None)
    for E0, W0 in starts:
        nrm2 = (E0 ** 2).sum((0, 2)) + (W0 ** 2).sum(0)
        sc = ((nrm2 ** (q / 2)).sum()) ** (-1 / q)
        E0, W0 = E0 * sc, W0 * sc
        x0 = np.concatenate([GF.pack(E0, W0), [0.0]])
        res = minimize(lambda x: -x[-1], x0,
                       jac=lambda x: np.concatenate([np.zeros(x.size - 1), [-1.0]]),
                       method="SLSQP",
                       constraints=[{"type": "ineq", "fun": lambda x: GF.margin_cons(x, p, n, m, idx, y)[0],
                                     "jac": lambda x: GF.margin_cons(x, p, n, m, idx, y)[1]},
                                    {"type": "eq", "fun": lambda x: GF.pen_cons(x, p, n, m, q)[0],
                                     "jac": lambda x: GF.pen_cons(x, p, n, m, q)[1]}],
                       options={"maxiter": 300, "ftol": 1e-12})
        E, W = GF.unpack(res.x, p, n, m)
        mg, _ = margin_of(E, W, idx, y, n)
        nrm2 = (E ** 2).sum((0, 2)) + (W ** 2).sum(0)
        Pq = float((nrm2 ** (q / 2)).sum())
        val = mg / Pq ** (nu / q)
        if np.isfinite(val) and val > best[0]:
            best = (val, (E.copy(), W.copy()))
    return best[1]


if __name__ == "__main__":
    main()
