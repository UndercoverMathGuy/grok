"""Ensemble cascade model of the GD-regime cleanup (init-agnostic test).

State: nf circuit amplitudes A_k = r_k^nu (nats, logit units, balanced
factors r_k) and one memorisation scaffold A_s = r_s^nu with uniform
coverage 1/c_s over wrong answers (train pairs only). Per train example the
margin against wrong offset D (1..p-1) is

    m(D) = A_s/c_s + sum_k A_k g_k(D),      g_k(D) = 1 - cos(2 pi k D / p)

CE = log(1 + sum_D e^{-m(D)}), p_D = e^{-m(D)} / (1 + sum e^{-m}).
Adam in the eps-dominated regime is GD with rate lr/eps on the loss and
decoupled wd lr*wd, so with the circuit path's margin-per-norm constant
absorbed into kappa = (lr/eps) * C^{2/nu}:

    r <- r (1 - lr wd) + kappa * G * r^{nu-1},   G_k = sum_D p_D g_k(D),
                                                 G_s = sum_D p_D / c_s.
kappa and c_s are MEASURED from runs without reference to K:
  kappa: final equilibrium of a grokked run, G_k * kappa * r_k^{nu-2} = lr wd
         for every surviving circuit (consistency: same kappa across k).
  c_s:   memorised state at cleanup start (train CE P0, margin m0 =
         log((p-1)/P0)): scaffold equilibrium (P0/c_s) kappa r_s^{nu-2} = lr wd
         with A_s = c_s m0.
Seeds: A_k(0) drawn iid from a generic ensemble (shape x scale), never
from data. Output: final K (same gap detector as the runs) per seed.
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cloud"))
from train_semifinal_torch import committee_from_coeffs


def coverage(p):
    k = np.arange(1, p // 2 + 1)[:, None]
    D = np.arange(1, p)[None, :]
    return 1.0 - np.cos(2 * np.pi * k * D / p)          # (nf, p-1)


def simulate(A0, m0, p, nu, lr, wd, eps, kappa, c_s, epochs, every=100,
             ceil=None):
    """A0: (S, nf) seed amplitudes per ensemble member; m0 scaffold margin.
    Returns final A (S, nf), A_s (S,), and trajectory list."""
    g = coverage(p)
    S, nf = A0.shape
    r = np.maximum(A0, 1e-12) ** (1.0 / nu)
    rs = np.full(S, (c_s * m0) ** (1.0 / nu))
    traj = []
    for t in range(epochs + 1):
        A = r ** nu
        As = rs ** nu
        m = As[:, None] / c_s + A @ g                     # (S, p-1)
        e = np.exp(-(m - m.min(1, keepdims=True)))        # stable softmax
        Z = np.exp(m.min(1, keepdims=True)) + e.sum(1, keepdims=True)
        pD = e / Z                                        # (S, p-1)
        Gk = pD @ g.T                                     # (S, nf)
        Gs = pD.sum(1) / c_s
        if t % every == 0:
            traj.append((t, A.copy(), As.copy(), float(np.log1p((pD.sum(1) / (1 - pD.sum(1))).mean()))))
        r = r * (1 - lr * wd) + kappa * Gk * r ** (nu - 1)
        rs = rs * (1 - lr * wd) + kappa * Gs * rs ** (nu - 1)
        if ceil is not None:
            r = np.minimum(r, ceil)
    return r ** nu, rs ** nu, traj


def simulate_pinned(A0, m0, p, nu, lr, wd, kappa, epochs, every=100,
                    greedy_seed=None):
    """'Pinned' scaffold: memorisation is repaired continuously so the worst
    train margin never drops below m0 (the data: train margin q05..q95 =
    15..17 nats throughout the takeover). Scaffold s(t) = max(0, m0 -
    min_D sum_k A_k g_k(D)), uniform over wrong answers. No c_s. The softmax
    drive is then always at floor scale, e^{-m0}, which multiplies kappa ~
    1/eps to an eps-free product: the model is eps-invariant by construction
    once m0 is expressed relative to log(1/eps)."""
    g = coverage(p)
    S, nf = A0.shape
    r = np.maximum(A0, 1e-12) ** (1.0 / nu)
    traj = []
    for t in range(epochs + 1):
        A = r ** nu
        cov = A @ g                                       # (S, p-1)
        s = np.maximum(0.0, m0 - cov.min(1, keepdims=True))
        m = s + cov
        e = np.exp(-(m - m.min(1, keepdims=True)))
        Z = np.exp(m.min(1, keepdims=True)) + e.sum(1, keepdims=True)
        pD = e / Z
        Gk = pD @ g.T
        if t % every == 0:
            traj.append((t, A.copy(), s[:, 0].copy(),
                         float(np.median(np.log1p(e.sum(1) / Z[:, 0] * np.exp(-m.min(1)) )))))
        r = r * (1 - lr * wd) + kappa * Gk * r ** (nu - 1)
    return r ** nu, traj


def calibrate_kappa(A_final, ce_final, p, nu, lr, wd, eps):
    """From a grokked run's final amplitudes (nats) and train CE floor."""
    g = coverage(p)
    m = A_final @ g
    e = np.exp(-m); Z = 1 + e.sum(); pD = e / Z
    P = e.sum() / Z
    Gk = pD @ g.T
    live = A_final >= 0.1 * A_final.max()
    r = A_final ** (1.0 / nu)
    kap = lr * wd / (Gk[live] * r[live] ** (nu - 2))
    return kap, P


def calibrate_cs(kappa, P0, p, nu, lr, wd):
    """Scaffold at its own floor: (P0/c_s) kappa (c_s m0)^{(nu-2)/nu} = lr wd."""
    m0 = math.log((p - 1) / P0)
    # c_s^{(nu-2)/nu - 1} = c_s^{-2/nu}  ->  c_s = (P0 kappa m0^{(nu-2)/nu} / (lr wd))^{nu/2}
    return (P0 * kappa * m0 ** ((nu - 2) / nu) / (lr * wd)) ** (nu / 2), m0


def draw_seed(shape, nf, S, rng):
    if shape == "exp":
        return rng.exponential(1.0, (S, nf))
    if shape == "absgauss":
        return np.abs(rng.normal(0, 1, (S, nf)))
    if shape == "lognorm1":
        return rng.lognormal(0, 1.0, (S, nf))
    if shape == "lognorm2":
        return rng.lognormal(0, 2.0, (S, nf))
    if shape == "uniform":
        return rng.uniform(0, 1, (S, nf))
    if shape == "pareto":
        return rng.pareto(2.0, (S, nf)) + 1
    if shape == "flat":
        return np.ones((S, nf)) * (1 + 1e-3 * rng.normal(0, 1, (S, nf)))
    raise ValueError(shape)


def khist(Afin):
    Ks, Kt = [], []
    for a in Afin:
        Ks.append(len(committee_from_coeffs(a)))
        Kt.append(int((a >= 0.1 * a.max()).sum()))
    h = lambda v: {int(k): int(c) for k, c in zip(*np.unique(v, return_counts=True))}
    return h(Ks), h(Kt), float(np.mean(Ks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=113)
    ap.add_argument("--nu", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--kappa", type=float, required=True)
    ap.add_argument("--cs", type=float, required=True)
    ap.add_argument("--m0", type=float, required=True)
    ap.add_argument("--epochs", type=int, default=50000)
    ap.add_argument("--S", type=int, default=48)
    ap.add_argument("--shapes", default="exp,absgauss,lognorm1,lognorm2,uniform,pareto,flat")
    ap.add_argument("--scales", default="0.01,0.03,0.1,0.3,1,3")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    nf = args.p // 2
    rng = np.random.default_rng(0)
    res = {}
    print(f"p={args.p} nu={args.nu} kappa={args.kappa:.3e} c_s={args.cs:.3e} m0={args.m0:.2f}")
    print("shape      scale   K hist (gap)                 K hist (thr)               meanK  final CE")
    for shape in args.shapes.split(","):
        for sc in [float(s) for s in args.scales.split(",")]:
            X = draw_seed(shape, nf, args.S, rng)
            A0 = sc * X / X.mean(1, keepdims=True)         # scale = mean seed amplitude (nats)
            Af, As, traj = simulate(A0, args.m0, args.p, args.nu, args.lr, args.wd,
                                    args.eps, args.kappa, args.cs, args.epochs)
            hg, ht, mk = khist(Af)
            res[f"{shape}@{sc}"] = dict(gap=hg, thr=ht, meanK=mk, ce=traj[-1][3],
                                        scaffold_final=float(np.median(As)))
            print(f"{shape:9s} {sc:6.2f}   {str(hg):28s} {str(ht):26s} {mk:5.2f}  {traj[-1][3]:.1e}  scaffold {np.median(As)/args.cs:.2f} nats")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
