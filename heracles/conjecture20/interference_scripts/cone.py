# Single-output-frequency gadget on alphabet Z_p with a cone of half-angle alpha:
#   Phi(N) = sum_i W_i prod_a h_i(a)^{N_a},  h_i(a) = u_i(a) rho^a,  u_i real unit vector in R^p
#   cone margin c = min_N ( Re Phi - |Im Phi| cot(alpha) );  efficiency = c / (B_p P^{nu/2}) / nu^{1-s}   (b = 2)
#   At alpha = pi/p this is exactly gamma/gamma_IF whenever K*(s,p) = 1.
import math, sys, numpy as np, torch
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..'))
from famp import counts
torch.set_default_dtype(torch.float64)


class Cone:
    def __init__(self, p, n, alpha, b=2.0):
        self.p, self.n, self.nu, self.b = p, n, n + 1, b
        self.N, _, _ = counts(n, p)
        self.Nc = self.N.to(torch.complex128)
        self.cot = 1 / math.tan(alpha)
        nu = self.nu; s = nu / b
        self.lref = (1 - s) * math.log(nu) + math.log(2) - (nu / 2) * math.log(2 * p * nu)
        ang = 2 * math.pi * torch.arange(p) / p
        self.rho = torch.exp(1j * ang.to(torch.complex128))

    def Phi(self, U, Wr, Wi):
        p, n, nu, b = self.p, self.n, self.nu, self.b
        Un = U / U.norm(dim=1, keepdim=True)
        W = torch.complex(Wr, Wi)
        absW = W.abs() + 1e-300
        lc = (b / 2) * (math.log(nu) + (2 / nu) * (0.5 * math.log(p / 2) + torch.log(absW)))
        logP = torch.logsumexp(lc, 0)
        lognorm = (nu / b) * logP + self.lref
        logh = torch.log(Un.to(torch.complex128) * self.rho[None, :])
        E = self.Nc @ logh.T
        return torch.exp(E + torch.log(W)[None, :] - lognorm).sum(1)

    def margins(self, U, Wr, Wi):
        Phi = self.Phi(U, Wr, Wi)
        return Phi.real - Phi.imag.abs() * self.cot


def optimise(cone, U, Wr, Wi, stages=(0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 3e-4, 1e-4), iters=800):
    P = [torch.tensor(np.array(x, dtype=float)).clone().requires_grad_() for x in (U, Wr, Wi)]
    for tau in stages:
        o = torch.optim.LBFGS(P, lr=1, max_iter=iters, line_search_fn="strong_wolfe",
                              tolerance_grad=1e-15, tolerance_change=1e-18)
        def closure():
            o.zero_grad(); g = cone.margins(*P)
            loss = tau * torch.logsumexp(-g / tau, 0); loss.backward(); return loss
        try:
            o.step(closure)
        except Exception:
            pass
    with torch.no_grad():
        r = cone.margins(*P).min().item()
    return r, [x.detach().numpy() for x in P]


def if_gadget(p, n):
    nu = n + 1; a = np.arange(p); al = np.pi * np.arange(nu) / nu
    U = np.array([np.cos(2 * np.pi * a / p + t) for t in al])
    W = np.exp(1j * n * al)
    return U, W.real.copy(), W.imag.copy()
