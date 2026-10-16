# Single-frequency single-feature networks on Z_p (slot-symmetric): u_i(a) = kappa_i + 2 cos(2 pi a/p + phi_i)
# params per neuron: kappa, phi, Wr, Wi.  Efficiency = cone margin / IF (cone.py normalisation).
import sys, math, time, numpy as np, torch
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from cone import Cone
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)


def margins(cone, kap, phi, Wr, Wi):
    a = torch.arange(cone.p, dtype=torch.float64)
    U = kap[:, None] + 2 * torch.cos(2 * math.pi * a[None, :] / cone.p + phi[:, None])
    return cone.margins(U, Wr, Wi)


def opt(cone, x, stages, iters=600):
    m = len(x) // 4
    P = torch.tensor(x).clone().requires_grad_()
    for tau in stages:
        o = torch.optim.LBFGS([P], lr=1, max_iter=iters, line_search_fn="strong_wolfe", tolerance_grad=1e-15, tolerance_change=1e-18)
        def closure():
            o.zero_grad(); g = margins(cone, P[:m], P[m:2*m], P[2*m:3*m], P[3*m:])
            loss = tau * torch.logsumexp(-g / tau, 0); loss.backward(); return loss
        try: o.step(closure)
        except Exception: pass
    with torch.no_grad():
        return margins(cone, P[:m], P[m:2*m], P[2*m:3*m], P[3*m:]).min().item(), P.detach().numpy()


def from_atoms(n, beta, theta, kap0=0.0):
    # beta_i = W_i e^{-i n phi_i}, xi_i = e^{2 i phi_i}  =>  phi = theta/2, W = beta e^{i n phi}
    phi = theta / 2
    W = beta * np.exp(1j * n * phi)
    m = len(beta)
    return np.concatenate([np.full(m, kap0), phi, W.real, W.imag])


if __name__ == "__main__":
    p, n = int(sys.argv[1]), int(sys.argv[2])
    cone = Cone(p, n, math.pi / p)
    X = np.load(sys.argv[3]); beta = X[0] + 1j * X[1]; theta = X[2]
    x = from_atoms(n, beta, theta)
    m = len(beta)
    r0 = margins(cone, *[torch.tensor(v) for v in (x[:m], x[m:2*m], x[2*m:3*m], x[3*m:])]).min().item()
    print("p=%d n=%d atoms as network (no DC): efficiency %.5f" % (p, n, r0), flush=True)
    for stages in [(1e-3, 3e-4, 1e-4), (0.03, 0.01, 0.003, 1e-3, 3e-4, 1e-4)]:
        r, y = opt(cone, x, stages)
        print("  after optimisation (stages %s): %.5f  kappa range [%.3f, %.3f]" % (stages[0], r, y[:m].min(), y[:m].max()), flush=True)
