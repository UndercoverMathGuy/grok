# Same structured slot-symmetric family as fam.py, but with general exponent b in the L_{2,b} penalty.
#   each neuron balanced: ||omega_i||^2 = nu * lam_i^2,  lam_i^2 = (sqrt(2/3)|v_i|)^{2/nu} (6+3k_i^2)^{n/nu}
#   P_b = sum_i (nu lam_i^2)^{b/2};  ratio = min_N conemargin / (gamma_IF(b) * P_b^{nu/b})
#   gamma_IF(b) = nu^{1-s} * 2 (6 nu)^{-nu/2} * 3/2,  s = nu/b   (p = 3, Gamma_s(3) = 3/2)
import torch, math, numpy as np, sys
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from fam import counts
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)


def ratio_b(params, m, n, Nc, b):
    nu = n + 1; s = nu / b
    phi, kap = params[:m], params[m:2 * m]
    vr, vi = params[2 * m:3 * m], params[3 * m:4 * m]
    w = torch.tensor([0.0, 2 * math.pi / 3, 4 * math.pi / 3])
    z = torch.exp(1j * (-phi[:, None] - w[None, :]).to(torch.complex128))
    q = 1 + kap[:, None] * z + z * z
    L = Nc @ torch.log(q.abs() + 1e-300).T; A = Nc @ torch.angle(q).T
    logv = torch.log(torch.sqrt(vr ** 2 + vi ** 2) + 1e-300); argv = torch.atan2(vi, vr)
    loglam2 = (2 / nu) * (0.5 * math.log(2 / 3) + logv) + (n / nu) * torch.log(6 + 3 * kap ** 2)
    logPb = torch.logsumexp((b / 2) * (math.log(nu) + loglam2), 0)
    logGIF = (1 - s) * math.log(nu) + math.log(2) - (nu / 2) * math.log(6 * nu) + math.log(1.5)
    lognorm = (nu / b) * logPb + logGIF
    mag = torch.exp(L + logv[None, :] - lognorm)
    Phi = (mag * torch.exp(1j * (A + argv[None, :]))).sum(1)
    return torch.minimum(Phi.real - Phi.imag / math.sqrt(3), Phi.real + Phi.imag / math.sqrt(3))


def optimise_b(n, m, b, init, stages=(0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001), iters=1500):
    Nc = counts(n)
    params = torch.tensor(init).clone().requires_grad_()
    for kappa in stages:
        o = torch.optim.LBFGS([params], lr=1, max_iter=iters, line_search_fn="strong_wolfe",
                              tolerance_grad=1e-16, tolerance_change=1e-18)

        def closure():
            o.zero_grad(); g = ratio_b(params, m, n, Nc, b)
            loss = kappa * torch.logsumexp(-g / kappa, 0); loss.backward(); return loss
        try:
            o.step(closure)
        except Exception:
            pass
    with torch.no_grad():
        r = ratio_b(params, m, n, Nc, b).min().item()
    return r, params.detach().numpy().copy()


if __name__ == "__main__":
    f, n, m = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    bs = [float(x) for x in sys.argv[4].split(',')]
    p0 = np.load(f)
    Nc = counts(n)
    prev = p0
    for b in bs:
        with torch.no_grad():
            r_same = ratio_b(torch.tensor(p0), m, n, Nc, b).min().item()
        r_opt, prev = optimise_b(n, m, b, prev)
        print("n=%d m=%d b=%.3f   same network: %.5f   re-optimised: %.5f" % (n, m, b, r_same, r_opt), flush=True)
