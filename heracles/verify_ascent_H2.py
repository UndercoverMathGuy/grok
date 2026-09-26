"""VERIFY link 5, second and stronger attack: ascent over the FULL (H) class.

Why a second script.  `verify_ascent_H.py` (and the inherited `gopt3_chain.py` V4) explore (H) by
GAUGE SYMMETRISATION, i.e. by the image of the map theta -> average of theta over the n+1 gauge
phases.  That map is NOT surjective onto (H): an (H) network need not be anybody's gauge average.
So the inherited evidence for Theorem D / the claim covers a proper subset of the hypothesis class.

Here the whole (H) class is searched directly: maximise

    ratio(theta) = gamma_q(theta) / [ (n+1)^{-(nu-q)/q} B Gamma_{nu/q}(p) ]
    subject to    J_{k,T} = 0  for all k in H, T non-empty,

over RAW weights (E, W) with no structure at all, by an augmented-Lagrangian / penalty ascent
(lambda annealed up by factors of 10) on the annealed soft-min margin.  The block is excluded from
every start.  The reported number is the exact brute-force margin of the final weights, and the
achieved same-|k| junk is reported alongside so the reader can judge whether (H) was really met.
"""
import itertools, json, os, sys
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "VERIFY")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_network import Gamma_s, margin_of, Pq, fourier, clean_and_junk, rebalance   # noqa


def junk_torch(E, W, p, n):
    """sum over k in H, T non-empty of |J_{k,T}|^2, normalised by sum_k |G_k|^2."""
    nf = (p - 1) // 2
    Eh = torch.fft.fft(E, dim=2)[:, :, 1:nf + 1]        # (n,m,nf) complex
    Wh = torch.fft.fft(W.T, dim=1)[:, 1:nf + 1]         # (m,nf)
    c = torch.prod(Eh, dim=0) * torch.conj(Wh)          # (m,nf)
    v = torch.conj(Eh) / (Eh + 1e-300)                  # (n,m,nf)
    G = c.sum(0)
    tot = 0.0
    for size in range(1, n + 1):
        for T in itertools.combinations(range(n), size):
            J = (c * torch.prod(v[list(T)], dim=0)).sum(0)
            tot = tot + (J.abs() ** 2).sum()
    return tot / ((G.abs() ** 2).sum() + 1e-300)


def ratio_torch(E, W, p, n, q, idx, s_row, tgt, beta):
    nu = n + 1
    h = torch.ones(idx.shape[0], E.shape[1])
    for j in range(n):
        h = h * E[j][:, idx[:, j]].T
    z = h @ W.T
    zc = z.gather(1, s_row.view(-1, 1))
    gaps = zc - z
    mask = torch.ones_like(gaps, dtype=torch.bool)
    mask.scatter_(1, s_row.view(-1, 1), False)
    soft = -torch.logsumexp(-beta * gaps[mask], dim=0) / beta
    nrm2 = (E ** 2).sum(dim=(0, 2)) + (W ** 2).sum(dim=0)
    P = (nrm2.clamp_min(1e-300) ** (q / 2)).sum()
    return soft / P ** (nu / q) / tgt


def start(p, n, m, seed, rng):
    kind = seed % 5
    E0 = rng.standard_normal((n, m, p)); W0 = rng.standard_normal((p, m))
    if kind == 1:                      # one neuron per frequency: MAXIMAL same-|k| junk
        E0 *= 0; W0 *= 0
        for i in range(m):
            k = 1 + (i % ((p - 1) // 2)); ph = rng.uniform(0, 2 * np.pi, n + 1)
            for j in range(n):
                E0[j, i] = np.cos(2 * np.pi * k * np.arange(p) / p + ph[j])
            W0[:, i] = np.cos(2 * np.pi * k * np.arange(p) / p + ph[n])
        E0 += 0.1 * rng.standard_normal(E0.shape); W0 += 0.1 * rng.standard_normal(W0.shape)
    elif kind == 2:                    # deliberately impure: three frequencies per neuron
        E0 *= 0.1; W0 *= 0.1
        for i in range(m):
            for k in rng.choice(np.arange(1, (p - 1) // 2 + 1), size=min(3, (p - 1) // 2),
                                replace=False):
                ph = rng.uniform(0, 2 * np.pi, n + 1)
                for j in range(n):
                    E0[j, i] += np.cos(2 * np.pi * k * np.arange(p) / p + ph[j])
                W0[:, i] += np.cos(2 * np.pi * k * np.arange(p) / p + ph[n])
    elif kind == 3:                    # clustered (deliberately non-optimal) support
        E0 *= 0; W0 *= 0
        for i in range(m):
            k = 1 + (i % min(3, (p - 1) // 2)); ph = rng.uniform(0, 2 * np.pi, n + 1)
            for j in range(n):
                E0[j, i] = np.cos(2 * np.pi * k * np.arange(p) / p + ph[j])
            W0[:, i] = np.cos(2 * np.pi * k * np.arange(p) / p + ph[n])
        E0 += 0.3 * rng.standard_normal(E0.shape); W0 += 0.3 * rng.standard_normal(W0.shape)
    elif kind == 4:
        E0 *= 3.0; W0 *= 3.0
    return E0, W0


def main():
    rep = []
    print(f"{'p':>4}{'n':>3}{'q':>5}{'m':>4} | {'Gamma_s':>12} {'final junk':>12} "
          f"{'best ratio (F=1 CLAIM)':>24}")
    for (p, n, m, q) in [(7, 2, 6, 2.0), (11, 2, 9, 2.0), (13, 2, 9, 2.0),
                         (11, 2, 9, 2.5), (5, 3, 8, 2.0)]:
        nu = n + 1
        Gs, _ = Gamma_s(p, nu / q, restarts=120, seed=p * 7 + n)
        B = 2 * (2.0 * p * nu) ** (-nu / 2.0)
        tgt = nu ** (-(nu - q) / q) * B * Gs
        idx = torch.tensor(np.array(list(itertools.product(range(p), repeat=n))))
        s_row = (idx.sum(1) % p)
        best = -np.inf; bestj = None; bw = None
        for seed in range(10):
            rng = np.random.default_rng(1000 * p + 10 * n + seed)
            E0, W0 = start(p, n, m, seed, rng)
            E = torch.tensor(E0, requires_grad=True); W = torch.tensor(W0, requires_grad=True)
            for beta, lam in [(20.0, 1e1), (100.0, 1e2), (500.0, 1e3),
                              (2000.0, 1e5), (8000.0, 1e7), (8000.0, 1e9)]:
                opt = torch.optim.LBFGS([E, W], max_iter=150, line_search_fn="strong_wolfe",
                                        tolerance_grad=1e-14, tolerance_change=1e-16)

                def closure():
                    opt.zero_grad()
                    loss = -ratio_torch(E, W, p, n, q, idx, s_row, tgt, beta) \
                        + lam * junk_torch(E, W, p, n)
                    loss.backward()
                    return loss
                try:
                    opt.step(closure)
                except Exception:
                    break
            with torch.no_grad():
                En, Wn = E.numpy().copy(), W.numpy().copy()
            En, Wn = rebalance(En, Wn, n)
            Eh, Wh = fourier(En, Wn, p, n)
            G, J = clean_and_junk(Eh, Wh, p, n)
            sc = max(np.abs(G).max(), 1e-300)
            jr = max(abs(v) for v in J.values()) / sc
            r = margin_of(En, Wn, p, n) / Pq(En, Wn, q) ** (nu / q) / tgt
            if jr < 1e-6 and r > best:                 # only count runs that really met (H)
                best, bestj, bw = r, jr, (En, Wn)
        print(f"{p:>4}{n:>3}{q:>5}{m:>4} | {Gs:>12.8f} {(bestj if bestj else float('nan')):>12.2e} "
              f"{best:>24.10f}")
        rep.append(dict(p=p, n=n, q=q, m=m, Gamma=Gs, junk=bestj, ratio=best))
        if best > 1.0:
            np.savez(os.path.join(OUT, f"VIOLATION2_p{p}_n{n}_q{q}.npz"), E=bw[0], W=bw[1])
            print("   *** ratio > 1 ***")
    mx = max(r["ratio"] for r in rep)
    print(f"\nmax over all cells = {mx:.10f} -> "
          f"{'(T) NOT broken on the full (H) class' if mx <= 1 else '*** (T) BROKEN ***'}")
    with open(os.path.join(OUT, "ascent_H2.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)
    print("wrote", os.path.join(OUT, "ascent_H2.json"))


if __name__ == "__main__":
    main()
