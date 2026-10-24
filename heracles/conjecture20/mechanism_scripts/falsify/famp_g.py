# Group-symmetric product networks at p (slots split into groups; slots in a group share their input vector).
#   neuron i: input u_{ik} in R^p for group k (unit), output w_i (unit), amplitude a_i = exp(la_i)
#   f(N)[c] = sum_i a_i w_i(c) prod_k prod_a u_{ik}(a)^{N^{(k)}_a};  balanced cost nu a_i^{2/nu};  b=2.
#   ratio = h / P^{nu/2} / gamma_IF,  gamma_IF = nu^{1-s} B_p Gamma_s(p),  s = nu/2.
import itertools, math, sys, os, numpy as np, torch
sys.path.insert(0, '../..')
from famp import counts, log_gamma_if
torch.set_default_dtype(torch.float64)

class GNet:
    def __init__(self, p, sizes, outfreq1=True):
        self.p, self.sizes = p, sizes; self.n = sum(sizes); self.nu = self.n + 1
        Ns = [counts(nk, p)[0] for nk in sizes]
        # cartesian product of per-group count vectors
        idx = np.array(list(itertools.product(*[range(len(N)) for N in Ns])))
        self.Nk = [Ns[k][torch.tensor(idx[:, k])] for k in range(len(sizes))]      # each: C x p
        ar = torch.arange(p, dtype=torch.float64)
        self.y = (sum(N @ ar for N in self.Nk)).long() % p
        C = len(self.y)
        self.mask = torch.ones(C, p, dtype=torch.bool); self.mask[torch.arange(C), self.y] = False
        self.lgif = log_gamma_if(p, self.n, 2.0)
        self.outfreq1 = outfreq1
        a = np.arange(p)
        self.cs = torch.tensor(np.stack([np.cos(2 * np.pi * a / p), np.sin(2 * np.pi * a / p)]))   # 2 x p
    def W_of(self, Wp):
        return Wp @ self.cs if self.outfreq1 else Wp
    def gaps(self, U, Wp, la):
        """U: m x g x p, Wp: m x 2 (freq-1 coefficients) or m x p, la: m"""
        nu = self.nu
        W = self.W_of(Wp)
        Un = U / U.norm(dim=2, keepdim=True); Wn = W / W.norm(dim=1, keepdim=True)
        L = 0; S = 0
        for k, N in enumerate(self.Nk):
            L = L + N @ torch.log(Un[:, k, :].abs() + 1e-300).T
            S = S + N @ (Un[:, k, :] < 0).double().T
        sgn = 1 - 2 * torch.remainder(S, 2)
        lc = math.log(nu) + (2 / nu) * la
        logP = torch.logsumexp(lc, 0)
        lognorm = (nu / 2) * logP + self.lgif
        Z = sgn * torch.exp(L + la[None, :] - lognorm)
        f = Z @ Wn
        g = f.gather(1, self.y[:, None]) - f
        return g[self.mask]
    def ratio(self, U, Wp, la):
        with torch.no_grad():
            return self.gaps(torch.as_tensor(U), torch.as_tensor(Wp), torch.as_tensor(la)).min().item()

def optimise(net, U, Wp, la, stages=(0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 3e-4, 1e-4), iters=500):
    U = torch.tensor(U).clone().requires_grad_(); Wp = torch.tensor(Wp).clone().requires_grad_()
    la = torch.tensor(la).clone().requires_grad_()
    for tau in stages:
        o = torch.optim.LBFGS([U, Wp, la], lr=1, max_iter=iters, line_search_fn="strong_wolfe",
                              tolerance_grad=1e-15, tolerance_change=1e-18)
        def closure():
            o.zero_grad(); g = net.gaps(U, Wp, la)
            loss = tau * torch.logsumexp(-g / tau, 0); loss.backward(); return loss
        try: o.step(closure)
        except Exception: pass
    with torch.no_grad(): r = net.gaps(U, Wp, la).min().item()
    return r, U.detach().numpy(), Wp.detach().numpy(), la.detach().numpy()

def if_block(p, n, g, outfreq1=True):
    nu = n + 1; a = np.arange(p); al = np.pi * np.arange(nu) / nu
    U = np.array([[np.cos(2 * np.pi * a / p + t) for k in range(g)] for t in al])
    if outfreq1: Wp = np.array([[np.cos(n * t), -np.sin(n * t)] for t in al])   # cos(2pi c/p + n t)
    else: Wp = np.array([np.cos(2 * np.pi * a / p + n * t) for t in al])
    return U, Wp, np.zeros(nu)
