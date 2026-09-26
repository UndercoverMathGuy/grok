"""VERIFY link 5 (the live part): an adversarial margin ascent INSIDE the (H) class.

(H) = "no same-|k| junk": J_{k,T} = sum_i c_ik prod_{j in T} v_ijk = 0 for every positive
frequency k and every non-empty T subset of {1..n}.  This is the hypothesis of Theorem D and of
the claim under audit.

Construction of the class.  Gauge symmetrisation: take ANY gated chain (E,W) of width m, rotate
every positive-frequency Fourier coefficient of each embedding by e^{i phi} and of the readout by
e^{i n phi} (an orthogonal map on each real vector, so norms are untouched), and average over the
n+1 phases phi_l = pi l/(n+1).  The result is a width-(n+1)m chain, generally IMPURE and with
cross-frequency junk intact, but with all same-|k| junk exactly zero.  Every network in the class
arises this way up to the free per-neuron rebalancing.

TEST.  Maximise   ratio = gamma_q / [ (n+1)^{-(nu-q)/q} * B * Gamma_{nu/q}(p) ]
over the pre-symmetrisation parameters (E,W).  The CLAIM under audit says ratio <= 1 with equality
at the share-weighted block.  Theorem D as proved only gives ratio <= F(n,q) (1.000905 at nu=3,q=2).
The block is EXCLUDED from every start (random and structured starts only).

Torch autograd, float64, annealed soft-min, L-BFGS.  Every reported number is recomputed by an
independent exact brute-force loop.
"""
import itertools, json, os, sys
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "VERIFY")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_network import Gamma_s, margin_of, Pq, fourier, clean_and_junk, rebalance   # noqa


def sym_net_torch(E, W, p, n):
    """Gauge-symmetrised network, in torch, differentiable.  Returns (En, Wn) of width (n+1)m."""
    nu = n + 1
    nf = (p - 1) // 2
    Eh = torch.fft.fft(E, dim=2)          # (n,m,p) complex
    Wh = torch.fft.fft(W.T, dim=1)        # (m,p)
    Es, Ws = [], []
    for l in range(nu):
        phi = np.pi * l / nu
        rot_e = torch.ones(p, dtype=torch.complex128)
        rot_w = torch.ones(p, dtype=torch.complex128)
        for k in range(1, nf + 1):
            rot_e[k] = np.exp(1j * phi); rot_e[p - k] = np.exp(-1j * phi)
            rot_w[k] = np.exp(1j * n * phi); rot_w[p - k] = np.exp(-1j * n * phi)
        Es.append(torch.real(torch.fft.ifft(Eh * rot_e, dim=2)))
        Ws.append(torch.real(torch.fft.ifft(Wh * rot_w, dim=1)).T / nu)
    return torch.cat(Es, dim=1), torch.cat(Ws, dim=1)


def ratio_torch(E, W, p, n, q, idx, s_row, tgt, beta):
    nu = n + 1
    En, Wn = sym_net_torch(E, W, p, n)
    h = torch.ones(idx.shape[0], En.shape[1])
    for j in range(n):
        h = h * En[j][:, idx[:, j]].T
    z = h @ Wn.T                                   # (p^n, p)
    zc = z.gather(1, s_row.view(-1, 1))
    gaps = zc - z                                  # 0 at the correct class
    mask = torch.ones_like(gaps, dtype=torch.bool)
    mask.scatter_(1, s_row.view(-1, 1), False)
    g = gaps[mask]
    soft = -torch.logsumexp(-beta * g, dim=0) / beta
    nrm2 = (En ** 2).sum(dim=(0, 2)) + (Wn ** 2).sum(dim=0)
    P = (nrm2.clamp_min(1e-300) ** (q / 2)).sum()
    return soft / P ** (nu / q) / tgt


def ascend(p, n, q, m, seed, tgt, steps=(250, 250, 250, 250)):
    rng = np.random.default_rng(seed)
    idx = torch.tensor(np.array(list(itertools.product(range(p), repeat=n))))
    s_row = (idx.sum(1) % p)
    kind = seed % 4
    if kind == 0:                                    # plain random
        E0 = rng.standard_normal((n, m, p))
        W0 = rng.standard_normal((p, m))
    elif kind == 1:                                  # one neuron per frequency (junk-maximal)
        E0 = np.zeros((n, m, p)); W0 = np.zeros((p, m))
        for i in range(m):
            k = 1 + (i % ((p - 1) // 2))
            ph = rng.uniform(0, 2 * np.pi, n + 1)
            for j in range(n):
                E0[j, i] = np.cos(2 * np.pi * k * np.arange(p) / p + ph[j])
            W0[:, i] = np.cos(2 * np.pi * k * np.arange(p) / p + ph[n])
        E0 += 0.05 * rng.standard_normal(E0.shape); W0 += 0.05 * rng.standard_normal(W0.shape)
    elif kind == 2:                                  # deliberately impure: two frequencies per neuron
        E0 = np.zeros((n, m, p)); W0 = np.zeros((p, m))
        for i in range(m):
            for k in [1 + (i % 3), 1 + ((i + 2) % ((p - 1) // 2))]:
                ph = rng.uniform(0, 2 * np.pi, n + 1)
                for j in range(n):
                    E0[j, i] += np.cos(2 * np.pi * k * np.arange(p) / p + ph[j])
                W0[:, i] += np.cos(2 * np.pi * k * np.arange(p) / p + ph[n])
    else:                                            # clustered support
        E0 = np.zeros((n, m, p)); W0 = np.zeros((p, m))
        for i in range(m):
            k = 1 + (i % min(4, (p - 1) // 2))
            ph = rng.uniform(0, 2 * np.pi, n + 1)
            for j in range(n):
                E0[j, i] = np.cos(2 * np.pi * k * np.arange(p) / p + ph[j])
            W0[:, i] = np.cos(2 * np.pi * k * np.arange(p) / p + ph[n])
        E0 += 0.3 * rng.standard_normal(E0.shape); W0 += 0.3 * rng.standard_normal(W0.shape)
    E = torch.tensor(E0, requires_grad=True)
    W = torch.tensor(W0, requires_grad=True)
    for beta in [20.0, 100.0, 500.0, 3000.0]:
        opt = torch.optim.LBFGS([E, W], max_iter=200, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-14, tolerance_change=1e-16)

        def closure():
            opt.zero_grad()
            loss = -ratio_torch(E, W, p, n, q, idx, s_row, tgt, beta)
            loss.backward()
            return loss
        try:
            opt.step(closure)
        except Exception:
            break
    with torch.no_grad():
        En, Wn = sym_net_torch(E, W, p, n)
    return En.numpy(), Wn.numpy()


def main():
    rep = []
    print(f"{'p':>4}{'n':>3}{'q':>5}{'m':>4} | {'Gamma_s':>12} {'max same-|k| junk':>18} "
          f"{'best ratio (F=1 target)':>24} {'/F':>10}")
    Fv = {(2, 2.0): 1.000905, (3, 2.0): 1.030841, (2, 2.5): 1.000451, (3, 2.5): 1.022630}
    for (p, n, m, q) in [(7, 2, 3, 2.0), (11, 2, 4, 2.0), (13, 2, 4, 2.0),
                         (11, 2, 4, 2.5), (5, 3, 3, 2.0), (7, 3, 3, 2.0)]:
        nu = n + 1
        Gs, _ = Gamma_s(p, nu / q, restarts=120, seed=p * 100 + n)
        B = 2 * (2.0 * p * nu) ** (-nu / 2.0)
        tgt = nu ** (-(nu - q) / q) * B * Gs
        best = -np.inf; jmax = 0.0; bestw = None
        for seed in range(12):
            En, Wn = ascend(p, n, q, m, seed, tgt)
            En, Wn = rebalance(En, Wn, n)
            Eh, Wh = fourier(En, Wn, p, n)
            G, J = clean_and_junk(Eh, Wh, p, n)
            sc = max(np.abs(G).max(), 1e-300)
            jmax = max(jmax, max(abs(v) for v in J.values()) / sc)
            r = margin_of(En, Wn, p, n) / Pq(En, Wn, q) ** (nu / q) / tgt   # exact brute force
            if r > best:
                best, bestw = r, (En, Wn)
        F = Fv.get((n, q), 1.0)
        print(f"{p:>4}{n:>3}{q:>5}{m:>4} | {Gs:>12.8f} {jmax:>18.2e} {best:>24.10f} {best/F:>10.6f}")
        rep.append(dict(p=p, n=n, q=q, m=m, Gamma=Gs, junk=jmax, ratio=best))
        if best > 1.0:
            np.savez(os.path.join(OUT, f"VIOLATION_p{p}_n{n}_q{q}.npz"), E=bestw[0], W=bestw[1])
            print("    *** ratio > 1: violation candidate saved ***")
    mx = max(r["ratio"] for r in rep)
    print(f"\nmax ratio over all cells = {mx:.10f}  -> "
          f"{'(T) NOT broken on the (H) class' if mx <= 1 else '*** (T) BROKEN ***'}")
    with open(os.path.join(OUT, "ascent_H.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)
    print("wrote", os.path.join(OUT, "ascent_H.json"))


if __name__ == "__main__":
    main()
