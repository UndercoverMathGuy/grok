# Slot-symmetric product networks for general odd p and general b.
#   neuron i: input factor u_i in R^p (same in every slot), output w_i in R^p, amplitude a_i > 0
#   f(N)[c] = sum_i a_i w_i(c) prod_a u_i(a)^{N_a}  with unit vectors u_i, w_i  (N = count vector)
#   balanced cost of neuron i: nu * a_i^{2/nu}  (= ||omega_i||_2^2),   P_b = sum_i (nu a_i^{2/nu})^{b/2}
#   ratio = h / P_b^{nu/b} / gamma_IF,   gamma_IF = nu^{1-s} B_p Gamma_s(p),  s = nu/b.
import itertools, math, sys, os
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gamma import Gamma
torch.set_default_dtype(torch.float64)


def counts(n, p):
    out = []
    def rec(prefix, left, k):
        if k == p - 1:
            out.append(prefix + [left]); return
        for v in range(left + 1):
            rec(prefix + [v], left - v, k + 1)
    rec([], n, 0)
    N = torch.tensor(out, dtype=torch.float64)
    y = (N @ torch.arange(p, dtype=torch.float64)).long() % p
    mult = torch.tensor([math.factorial(n) / np.prod([math.factorial(int(v)) for v in row]) for row in out])
    return N, y, mult


def gamma_if(p, n, b):
    nu = n + 1; s = nu / b
    Bp = 2 * (2 * p * nu) ** (-nu / 2)
    return nu ** (1 - s) * Bp * Gamma(p, s)


def log_gamma_if(p, n, b):
    nu = n + 1; s = nu / b
    return (1 - s) * math.log(nu) + math.log(2) - (nu / 2) * math.log(2 * p * nu) + math.log(Gamma(p, s))


class Net:
    def __init__(self, p, n, b=2.0):
        self.p, self.n, self.b, self.nu = p, n, b, n + 1
        self.N, self.y, self.mult = counts(n, p)
        self.mask = torch.ones(len(self.y), p, dtype=torch.bool)
        self.mask[torch.arange(len(self.y)), self.y] = False
        self.lgif = log_gamma_if(p, n, b)

    def gaps(self, U, W, la):
        """normalised gaps / gamma_IF for every (count vector, wrong class)"""
        nu, b = self.nu, self.b
        Un = U / U.norm(dim=1, keepdim=True); Wn = W / W.norm(dim=1, keepdim=True)
        L = self.N @ torch.log(Un.abs() + 1e-300).T
        S = self.N @ (Un < 0).double().T
        sgn = 1 - 2 * torch.remainder(S, 2)
        lc = (b / 2) * (math.log(nu) + (2 / nu) * la)
        logP = torch.logsumexp(lc, 0)
        lognorm = (nu / b) * logP + self.lgif
        Z = sgn * torch.exp(L + la[None, :] - lognorm)
        f = Z @ Wn
        g = f.gather(1, self.y[:, None]) - f
        return g[self.mask]

    def ratio(self, U, W, la):
        return self.gaps(U, W, la).min().item()


def optimise(net, U, W, la, stages=(0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 3e-4, 1e-4), iters=1000):
    U = torch.tensor(U).clone().requires_grad_(); W = torch.tensor(W).clone().requires_grad_()
    la = torch.tensor(la).clone().requires_grad_()
    for tau in stages:
        o = torch.optim.LBFGS([U, W, la], lr=1, max_iter=iters, line_search_fn="strong_wolfe",
                              tolerance_grad=1e-15, tolerance_change=1e-18)
        def closure():
            o.zero_grad(); g = net.gaps(U, W, la)
            loss = tau * torch.logsumexp(-g / tau, 0); loss.backward(); return loss
        try:
            o.step(closure)
        except Exception:
            pass
    with torch.no_grad():
        r = net.gaps(U, W, la).min().item()
    return r, U.detach().numpy(), W.detach().numpy(), la.detach().numpy()


def if_block(p, n, j=1):
    nu = n + 1; a = np.arange(p)
    al = np.pi * np.arange(nu) / nu
    U = np.array([np.cos(2 * np.pi * j * a / p + t) for t in al])
    W = np.array([np.cos(2 * np.pi * j * a / p + n * t) for t in al])
    return U, W, np.zeros(nu)


def arc_init(p, n, m, span_deg, kap_mid, kap_end, slope, j=1, extra=0.0, rng=None):
    """p=3 mechanism transplanted: u_i = kappa_i + 2cos(2 pi j a/p + phi_i), output phase tied to n*phi_i."""
    a = np.arange(p)
    t = np.linspace(-0.5, 0.5, m) if m > 1 else np.zeros(1)
    phi = np.pi + math.radians(span_deg) * t
    kap = kap_mid + (kap_end - kap_mid) * (2 * t) ** 2
    v = np.exp(1j * slope * (phi - np.pi))
    U = np.array([k + 2 * np.cos(2 * np.pi * j * a / p + f) for k, f in zip(kap, phi)])
    Wc = v * np.exp(-1j * n * phi)
    W = np.array([np.real(wc * np.exp(2j * np.pi * j * a / p)) for wc in Wc])
    if extra and rng is not None:
        U = U + extra * rng.standard_normal(U.shape)
        W = W + extra * rng.standard_normal(W.shape)
    return U, W, np.zeros(m)
