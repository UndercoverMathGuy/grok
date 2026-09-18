"""relu_dense_check.py -- exact finite checks behind heracles/THEORY_relu.md (p = 113 grid, no training).

Every number here is an exact DFT on the real Z_p x Z_p grid of an explicit ReLU network, never a
continuum approximation. Parts:
  A  single-frequency ReLU neuron: diagonal Fourier series d_n = (4/pi^2)(-1)^(n+1)/(4n^2-1) and
     clean efficiency eta_1 = sqrt(2/pi^2 - 16/pi^4)
  B  effective-quadratic law: for an n-frequency or Gaussian dense neuron, the clean coefficient at k is
     u_hat(k) v_hat(k) / (sqrt(2 pi) R), R = rms of the pre-activation (dilution law)
  C  cross junk: the (k, k') coefficient of a locked multi-frequency neuron equals the (k, k) one
  D  sup over ALL neurons (u, v) of the clean efficiency eta = ||clean||/||x|| (gradient ascent, restarts)
  E  locking sweep: dense-locked population (overlap rho) vs the sparse population on the ladder supports,
     exact margins, exact nuclear norms, normalised margin at nu 3 and nu 4; rho = 1 with a search over
     per-frequency global phases
Usage: uv run python heracles/relu_dense_check.py --parts ABCDE --out heracles/relu_dense_check.json
"""
import argparse, json, math, time
from pathlib import Path

import numpy as np

P = 113
NF = (P - 1) // 2
ROOT = Path(__file__).resolve().parent
TH = 2 * np.pi * np.arange(P) / P
C_GAUSS = 1 / (2 * math.sqrt(2 * math.pi))       # |x| second-order constant; relu is half of it


def diag(H):
    return np.array([H[k, k] for k in range(1, NF + 1)])


def relu(x):
    return np.maximum(x, 0.0)


def table(u, v):
    return u[:, None] + v[None, :]


def fft2n(h):
    return np.fft.fft2(h) / P ** 2


def fft1n(f):
    return np.fft.fft(f) / P


def eta(u, v):
    """clean efficiency: sqrt(2 sum_k |H(k,k)|^2) / sqrt(Var x). Lemma 4 says <= 1."""
    x = table(u, v)
    d = diag(fft2n(relu(x)))
    return math.sqrt(2 * float(np.sum(np.abs(d) ** 2))) / math.sqrt(float(x.var()))


def cos_neuron(freqs, ru, rv, pu, pv):
    u = sum(r * np.cos(k * TH + f) for k, r, f in zip(freqs, ru, pu))
    v = sum(s * np.cos(k * TH + g) for k, s, g in zip(freqs, rv, pv))
    return u, v


# ------------------------------------------------------------------ A
def part_A():
    k = 7
    u, v = cos_neuron([k], [1.0], [1.0], [0.0], [0.0])
    x = table(u, v)
    H = fft2n(relu(x))
    rows = []
    for n in range(1, 7):
        kk = (n * k) % P
        meas = H[kk, kk].real
        exact = (4 / math.pi ** 2) * (-1) ** (n + 1) / (4 * n ** 2 - 1)
        rows.append(dict(n=n, k=kk, measured=meas, exact=exact))
    e = eta(u, v)
    e_exact = math.sqrt(2 / math.pi ** 2 - 16 / math.pi ** 4)
    # linear half has no diagonal content
    Hl = fft2n(x / 2)
    lin = float(np.max(np.abs(diag(Hl))))
    print("A. single-frequency relu neuron, x = cos(ka) + cos(kb), k = 7")
    for r in rows:
        print(f"   n={r['n']}  (nk,nk)={r['k']:3d}  measured {r['measured']:+.6f}  exact {r['exact']:+.6f}")
    print(f"   eta_1 measured {e:.5f}  exact sqrt(2/pi^2-16/pi^4) = {e_exact:.5f};  linear half diag content {lin:.1e}")
    return dict(harmonics=rows, eta1=e, eta1_exact=e_exact, linear_diag=lin)


# ------------------------------------------------------------------ B
def part_B(rng):
    S8 = json.loads((ROOT / "kstar.json").read_text())[str(P)]["S"]["8"]
    freqs_all = [k for k in range(1, NF + 1)]
    out = []
    print("B. effective-quadratic law: d_k = relu diag coeff at k;  pred = r_k s_k /(4 sqrt(2pi) R)")
    for n in (1, 2, 3, 4, 6, 8, 12, 20, 56):
        freqs = (S8 + [k for k in freqs_all if k not in S8])[:n]
        pu, pv = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)
        u, v = cos_neuron(freqs, [1.0] * n, [1.0] * n, pu, pv)
        x = table(u, v); R = math.sqrt(x.var())
        H = fft2n(relu(x))
        d = np.array([abs(H[k, k]) for k in freqs])
        pred = 1.0 / (4 * math.sqrt(2 * math.pi) * R)
        e = math.sqrt(2 * float(np.sum(np.abs(diag(H)) ** 2))) / R
        e_pred = 0.5 * C_GAUSS * math.sqrt(2.0 / n)            # eta_n -> 0.141/sqrt(n)
        out.append(dict(n=n, R=R, d_mean=float(d.mean()), d_min=float(d.min()), d_max=float(d.max()),
                        pred=pred, ratio=float(d.mean() / pred), eta=e, eta_pred=e_pred,
                        dil_vs_single=float(d.mean() / (4 / (3 * math.pi ** 2)))))
        print(f"   n={n:2d} R={R:.3f}  d_k mean {d.mean():.5f} [{d.min():.5f},{d.max():.5f}]  pred {pred:.5f}  "
              f"ratio {d.mean()/pred:.3f}   eta_n {e:.4f} (pred {e_pred:.4f})   d_n/d_1 {out[-1]['dil_vs_single']:.3f}"
              f"  x sqrt(n) = {out[-1]['dil_vs_single']*math.sqrt(n):.3f}")
    # Gaussian dense neurons: complex per-frequency comparison H(k,k) vs u_hat(k) v_hat(k)/(sqrt(2pi) R)
    gauss = []
    for seed in range(4):
        g = np.random.default_rng(100 + seed)
        u, v = g.standard_normal(P), g.standard_normal(P)
        x = table(u, v); R = math.sqrt(x.var())
        d = diag(fft2n(relu(x)))
        uh, vh = fft1n(u), fft1n(v)
        pred = np.array([uh[k] * vh[k] for k in range(1, NF + 1)]) / (math.sqrt(2 * math.pi) * R)
        corr = abs(np.vdot(pred, d)) / (np.linalg.norm(pred) * np.linalg.norm(d))
        slope = float((np.vdot(pred, d) / np.vdot(pred, pred)).real)
        resid = np.linalg.norm(d - slope * pred) / np.linalg.norm(d)
        e = math.sqrt(2 * float(np.sum(np.abs(d) ** 2))) / R
        quad = np.array([2 * uh[k] * vh[k] for k in range(1, NF + 1)])
        gauss.append(dict(seed=seed, R=R, corr=float(corr), slope=slope, resid=float(resid), eta=e,
                          relu_over_quad_pred=1 / (2 * math.sqrt(2 * math.pi) * R)))
        print(f"   gaussian dense seed {seed}: R={R:.3f}  corr(H_kk, u_hat v_hat/(sqrt(2pi)R)) = {corr:.4f}  "
              f"slope {slope:.3f}  resid {resid:.3f}  eta {e:.4f}")
    return dict(cosine=out, gaussian=gauss)


# ------------------------------------------------------------------ C
def part_C(rng):
    S8 = json.loads((ROOT / "kstar.json").read_text())[str(P)]["S"]["8"]
    print("C. cross junk of a locked multi-frequency neuron: |H(k,k')| vs |H(k,k)|")
    out = []
    for n in (2, 3, 4, 6):
        freqs = S8[:n]
        u, v = cos_neuron(freqs, [1.0] * n, [1.0] * n, [0.3] * n, [1.1] * n)
        H = fft2n(relu(table(u, v)))
        dg = np.array([abs(H[k, k]) for k in freqs])
        cr = np.array([abs(H[k, kp]) for k in freqs for kp in freqs if k != kp])
        # a k+-k' leakage entry for scale
        leak = max(abs(H[(freqs[0] + freqs[1]) % P, (freqs[0] + freqs[1]) % P]),
                   abs(H[(freqs[0] - freqs[1]) % P, (freqs[0] - freqs[1]) % P]))
        out.append(dict(n=n, diag_mean=float(dg.mean()), cross_mean=float(cr.mean()), cross_min=float(cr.min()),
                        cross_max=float(cr.max()), leak_k1pmk2=float(leak)))
        print(f"   n={n}: diag {dg.mean():.5f}  cross mean {cr.mean():.5f} [{cr.min():.5f},{cr.max():.5f}]  "
              f"ratio {cr.mean()/dg.mean():.3f}   diag leak at k1+-k2: {leak:.5f}")
    return out


# ------------------------------------------------------------------ D
def part_D(rng):
    import torch
    torch.manual_seed(0)
    print("D. sup over all neurons of the clean efficiency eta (and the l1 efficiency), gradient ascent")
    ks = torch.arange(1, NF + 1)

    def objective(u, v, l1=False):
        x = u[:, None] + v[None, :]
        h = torch.relu(x)
        H = torch.fft.fft2(h) / P ** 2
        d = H[ks, ks]
        if l1:
            return d.abs().sum() / torch.sqrt(x.var())
        return torch.sqrt(2 * (d.abs() ** 2).sum()) / torch.sqrt(x.var())

    res = {}
    for l1 in (False, True):
        best = (-1, None)
        starts = []
        for s in range(12):
            g = torch.Generator().manual_seed(s)
            if s < 6:
                u0, v0 = torch.randn(P, generator=g), torch.randn(P, generator=g)
            else:                                    # single-frequency + noise starts
                k = int(rng.integers(1, NF + 1)); th = torch.tensor(TH)
                u0 = torch.cos(k * th) + 0.3 * torch.randn(P, generator=g)
                v0 = torch.cos(k * th + 0.7) + 0.3 * torch.randn(P, generator=g)
            u, v = u0.clone().requires_grad_(True), v0.clone().requires_grad_(True)
            opt = torch.optim.Adam([u, v], lr=0.02)
            val = 0.0
            for it in range(2500):
                opt.zero_grad(); val = objective(u, v, l1); (-val).backward(); opt.step()
            val = float(objective(u, v, l1))
            starts.append(val)
            if val > best[0]:
                best = (val, (u.detach().numpy().copy(), v.detach().numpy().copy()))
        u, v = best[1]
        x = table(u, v); H = fft2n(relu(x)); d = np.abs(diag(H))
        uh = np.abs(fft1n(u))[1:NF + 1]; vh = np.abs(fft1n(v))[1:NF + 1]
        top = np.argsort(-d)[:4] + 1
        share2 = float(np.sum(np.sort(d ** 2)[-2:]) / np.sum(d ** 2))
        n5 = int(np.sum(d > 0.05 * d.max()))
        ushare = float(np.sort(uh ** 2)[-1] / np.sum(uh ** 2)); vshare = float(np.sort(vh ** 2)[-1] / np.sum(vh ** 2))
        name = "l1" if l1 else "l2"
        res[name] = dict(best=best[0], starts=starts, top_freqs=[int(t) for t in top], top2_share=share2,
                         n_freqs_5pct=n5, u_top_freq_share=ushare, v_top_freq_share=vshare)
        print(f"   {name} efficiency: best {best[0]:.5f}  starts {np.round(starts, 4).tolist()}")
        print(f"      maximiser: top clean freqs {top.tolist()}, top-2 share {share2:.3f}, #freqs>5% {n5}, "
              f"u/v energy in their top frequency {ushare:.3f}/{vshare:.3f}")
    return res


# ------------------------------------------------------------------ E
def margin_of(z):
    """z (p, p, p) indexed [c, a, b]; returns min margin and the clean-projection margin."""
    A, B = np.meshgrid(np.arange(P), np.arange(P), indexing="ij")
    Cc = (A + B) % P
    zy = z[Cc, A, B]
    zz = z.copy(); zz[Cc, A, B] = -np.inf
    m = float((zy - zz.max(0)).min())
    # clean projection g(D) = mean over a,b of z[a+b-D, a, b]
    g = np.array([z[(Cc - D) % P, A, B].mean() for D in range(P)])
    mc = float((g[0] - g[1:]).min())
    return m, mc, g


def clean_amp(g):
    gh = np.fft.fft(g) / P
    return np.abs(gh[1:NF + 1])


def build_logits(U, V, W, chunk=512):
    """U, V (m, p); W (p, m) -> z (p, p, p) = sum_i W[c,i] relu(U[i,a] + V[i,b])."""
    m = U.shape[0]
    z = np.zeros((P, P * P), dtype=np.float64)
    for s in range(0, m, chunk):
        h = relu(U[s:s + chunk, :, None] + V[s:s + chunk, None, :]).reshape(-1, P * P)
        z += W[:, s:s + chunk] @ h
    return z.reshape(P, P, P)


def nuc(M):
    return float(np.linalg.svd(M, compute_uv=False).sum())


def gamma(margin, X, W, nu):
    x = nuc(X)
    if nu == 3:
        return margin / (3 ** 1.5 * x * np.linalg.norm(W))
    return margin / (16 * x * nuc(W))


def sparse_pop(S, rep=4, grid=8):
    K = len(S)
    ph = 2 * np.pi * np.arange(grid) / grid
    PH, PS = np.meshgrid(ph, ph, indexing="ij"); PH, PS = PH.ravel(), PS.ravel()
    U, V, W = [], [], []
    for k in S:
        for _ in range(rep):
            U.append(np.cos(k * TH[None, :] + PH[:, None]))
            V.append(np.cos(k * TH[None, :] + PS[:, None]))
            W.append(np.cos(k * TH[None, :] + (PH + PS)[:, None]))
    U, V, W = np.vstack(U), np.vstack(V), np.vstack(W).T
    return U, V, W


def dense_pop(S, rho, rng, rep=32, grid=8, chi=None, chip=None):
    """m = grid^2 * rep neurons, every neuron carries all of S at amplitude 1/sqrt(K).
    Per-(neuron, frequency) jitter: locked to the base phase with prob q = sqrt(rho), else uniform on the
    4-grid => E e^{i theta} = q, pairwise overlap E e^{i(theta_k - theta_k')} = rho."""
    K = len(S); q = math.sqrt(rho)
    ph = 2 * np.pi * np.arange(grid) / grid
    PH, PS = np.meshgrid(ph, ph, indexing="ij")
    PH, PS = np.tile(PH.ravel(), rep), np.tile(PS.ravel(), rep)
    m = PH.size
    chi = np.zeros(K) if chi is None else chi
    chip = np.zeros(K) if chip is None else chip

    def jitter():
        lock = rng.uniform(size=(m, K)) < q
        j = (np.pi / 2) * rng.integers(0, 4, size=(m, K))
        return np.where(lock, 0.0, j)
    TU, TV = jitter(), jitter()
    U = np.zeros((m, P)); V = np.zeros((m, P)); W = np.zeros((m, P))
    for j, k in enumerate(S):
        U += np.cos(k * TH[None, :] + (PH + chi[j] + TU[:, j])[:, None])
        V += np.cos(k * TH[None, :] + (PS + chip[j] + TV[:, j])[:, None])
        W += np.cos(k * TH[None, :] + (PH + PS + chi[j] + chip[j] + TU[:, j] + TV[:, j])[:, None])
    return U / math.sqrt(K), V / math.sqrt(K), W.T / math.sqrt(K)


def f_overlap(rho, K):
    return (math.sqrt(1 + (K - 1) * rho) + (K - 1) * math.sqrt(1 - rho)) / K


def part_E(rng, Ks=(2, 3, 4, 5, 6, 8), rhos=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0), n_phase=300):
    lad = json.loads((ROOT / "kstar.json").read_text())[str(P)]
    out = {}
    print("E. locking sweep on the ladder supports (exact tables, exact nuclear norms)")
    for K in Ks:
        S = lad["S"][str(K)]
        t0 = time.time()
        U, V, W = sparse_pop(S, rep=max(1, 2048 // (64 * K)))
        X = np.vstack([U, V])
        z = build_logits(U, V, W)
        ms, mcs, gs = margin_of(z)
        As = clean_amp(gs)[[k - 1 for k in S]]
        row = dict(S=S, M_K=lad["M_K"][str(K)], sparse=dict(m=int(U.shape[0]), margin=ms, clean_margin=mcs,
                   junk_loss=1 - ms / mcs, amp_mean=float(As.mean()), X_nuc=nuc(X), X_fro=float(np.linalg.norm(X)),
                   W_nuc=nuc(W), W_fro=float(np.linalg.norm(W)),
                   gamma3=gamma(ms, X, W, 3), gamma4=gamma(ms, X, W, 4)), dense={})
        print(f"  K={K} S={S}  sparse m={U.shape[0]}: margin {ms:.4f} (clean {mcs:.4f}, junk loss {1-ms/mcs:.3f})  "
              f"||X||_*={nuc(X):.1f} ||X||_F={np.linalg.norm(X):.1f}  gamma3 {row['sparse']['gamma3']:.3e} "
              f"gamma4 {row['sparse']['gamma4']:.3e}   [{time.time()-t0:.0f}s]")
        g3s, g4s = row["sparse"]["gamma3"], row["sparse"]["gamma4"]
        xn0 = None
        for rho in rhos:
            rep = 1 if rho == 1.0 else 32
            Ud, Vd, Wd = dense_pop(S, rho, rng, rep=rep)
            Xd = np.vstack([Ud, Vd])
            zd = build_logits(Ud, Vd, Wd)
            md, mcd, gd = margin_of(zd)
            Ad = clean_amp(gd)[[k - 1 for k in S]]
            xn = nuc(Xd) / (Xd.shape[0] ** 0.5 if False else 1.0)
            # scale-free comparisons: nuclear norm per unit Frobenius
            xnf = nuc(Xd) / np.linalg.norm(Xd); wnf = nuc(Wd) / np.linalg.norm(Wd)
            if rho == 0.0:
                xn0 = xnf
            g3 = gamma(md, Xd, Wd, 3); g4 = gamma(md, Xd, Wd, 4)
            g3c = gamma(mcd, Xd, Wd, 3); g4c = gamma(mcd, Xd, Wd, 4)
            d = dict(rho=rho, m=int(Ud.shape[0]), margin=md, clean_margin=mcd, junk_loss=1 - md / mcd if mcd > 0 else None,
                     amp_mean=float(Ad.mean()), amp_vs_sparse_equalF=float(Ad.mean() / As.mean() *
                     (np.linalg.norm(X) / np.linalg.norm(Xd)) * (np.linalg.norm(W) / np.linalg.norm(Wd))),
                     Xnuc_per_fro=xnf, Xnuc_ratio_to_rho0=xnf / xn0, f_pred=f_overlap(rho, K), Wnuc_per_fro=wnf,
                     gamma3=g3, gamma4=g4, r3=g3 / g3s, r4=g4 / g4s, r3_clean=g3c / g3s, r4_clean=g4c / g4s)
            row["dense"][str(rho)] = d
            print(f"     rho={rho:<4} m={Ud.shape[0]:5d}: margin {md:+.4f} clean {mcd:.4f}  amp/sparse@equalF {d['amp_vs_sparse_equalF']:.3f}"
                  f"  ||X||_*/||X||_F {xnf:.2f} (ratio to rho0 {xnf/xn0:.3f}, f(rho) {f_overlap(rho,K):.3f})  "
                  f"g3/g3s {g3/g3s:+.3f} g4/g4s {g4/g4s:+.3f}   junk-free: {g3c/g3s:.3f} / {g4c/g4s:.3f}")
        # rho = 1 with a search over per-frequency global phases (m = 64 suffices: all replicas identical)
        best = (-np.inf, None); t0 = time.time()
        cands = [(np.zeros(K), np.zeros(K))] + [(rng.uniform(0, 2 * np.pi, K), rng.uniform(0, 2 * np.pi, K))
                                                 for _ in range(n_phase)]
        for chi, chip in cands:
            Ud, Vd, Wd = dense_pop(S, 1.0, rng, rep=1, chi=chi, chip=chip)
            md, mcd, _ = margin_of(build_logits(Ud, Vd, Wd))
            if md > best[0]:
                best = (md, (chi, chip, mcd, gamma(md, np.vstack([Ud, Vd]), Wd, 3), gamma(md, np.vstack([Ud, Vd]), Wd, 4)))
        chi, chip = best[1][0], best[1][1]
        for it in range(200):                      # local refinement
            c2, cp2 = chi + rng.normal(0, 0.3, K), chip + rng.normal(0, 0.3, K)
            Ud, Vd, Wd = dense_pop(S, 1.0, rng, rep=1, chi=c2, chip=cp2)
            md, mcd, _ = margin_of(build_logits(Ud, Vd, Wd))
            if md > best[0]:
                best = (md, (c2, cp2, mcd, gamma(md, np.vstack([Ud, Vd]), Wd, 3), gamma(md, np.vstack([Ud, Vd]), Wd, 4)))
                chi, chip = c2, cp2
        row["dense"]["1.0_phase_search"] = dict(margin=best[0], clean_margin=best[1][2], r3=best[1][3] / g3s,
                                                r4=best[1][4] / g4s, n_tried=n_phase + 201)
        print(f"     rho=1 best over {n_phase+201} global-phase choices: margin {best[0]:+.4f} (clean {best[1][2]:.4f})"
              f"  g3/g3s {best[1][3]/g3s:+.3f}  g4/g4s {best[1][4]/g4s:+.3f}   [{time.time()-t0:.0f}s]")
        out[str(K)] = row
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="ABCDE")
    ap.add_argument("--out", default=str(ROOT / "relu_dense_check.json"))
    ap.add_argument("--n_phase", type=int, default=300)
    args = ap.parse_args()
    rng = np.random.default_rng(20260916)
    res = {}
    if "A" in args.parts: res["A"] = part_A()
    if "B" in args.parts: res["B"] = part_B(rng)
    if "C" in args.parts: res["C"] = part_C(rng)
    if "D" in args.parts: res["D"] = part_D(rng)
    if "E" in args.parts: res["E"] = part_E(rng, n_phase=args.n_phase)
    Path(args.out).write_text(json.dumps(res, indent=1, default=float))
    print("wrote", args.out)
