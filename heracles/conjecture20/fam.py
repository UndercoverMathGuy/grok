# Structured slot-symmetric family (p=3, b=2):
#   Phi(N) = sum_i v_i prod_a q_i(zeta_i w^{-a})^{N_a},  q_k(z) = 1 + k z + z^2,  zeta_i = e^{-i phi_i}
#   cost P = nu * sum_i (sqrt(2/3)|v_i|)^{2/nu} (6 + 3 k_i^2)^{n/nu}
#   ratio = min_N conemargin(Phi(N)) / (gamma_IF * P^{nu/2})
import torch, math, numpy as np
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)


def counts(n):
    return torch.tensor([(a, b, n - a - b) for a in range(n + 1) for b in range(n + 1 - a)], dtype=torch.float64)


def ratio(params, m, n, Nc):
    nu = n + 1
    phi, kap = params[:m], params[m:2 * m]
    vr, vi = params[2 * m:3 * m], params[3 * m:4 * m]
    w = torch.tensor([0.0, 2 * math.pi / 3, 4 * math.pi / 3])
    ang = -phi[:, None] - w[None, :]
    z = torch.exp(1j * ang.to(torch.complex128))
    q = 1 + kap[:, None] * z + z * z
    logabs = torch.log(q.abs() + 1e-300); arg = torch.angle(q)
    L = Nc @ logabs.T; A = Nc @ arg.T
    logv = torch.log(torch.sqrt(vr ** 2 + vi ** 2) + 1e-300); argv = torch.atan2(vi, vr)
    lc = (2 / nu) * (0.5 * math.log(2 / 3) + logv) + (n / nu) * torch.log(6 + 3 * kap ** 2)
    logP = math.log(nu) + torch.logsumexp(lc, 0)
    logGIF = math.log(3 * nu) - nu / 2 * math.log(6 * nu * nu)
    lognorm = (nu / 2) * logP + logGIF
    mag = torch.exp(L + logv[None, :] - lognorm)
    Phi = (mag * torch.exp(1j * (A + argv[None, :]))).sum(1)
    return torch.minimum(Phi.real - Phi.imag / math.sqrt(3), Phi.real + Phi.imag / math.sqrt(3))


def optimise(n, m, init, stages=(0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001), iters=1500):
    Nc = counts(n)
    params = torch.tensor(init).clone().requires_grad_()
    for kappa in stages:
        o = torch.optim.LBFGS([params], lr=1, max_iter=iters, line_search_fn="strong_wolfe",
                              tolerance_grad=1e-16, tolerance_change=1e-18)

        def closure():
            o.zero_grad(); g = ratio(params, m, n, Nc)
            loss = kappa * torch.logsumexp(-g / kappa, 0); loss.backward(); return loss
        try:
            o.step(closure)
        except Exception:
            pass
    with torch.no_grad():
        r = ratio(params, m, n, Nc).min().item()
    return r, params.detach().numpy().copy()


def structured_init(m, span_deg=None, kap_mid=0.5, kap_end=0.7, slope=-1.1, jitter=0.0, rng=None):
    span = math.radians(span_deg) if span_deg else (2 * math.pi / 3) * (m - 1) / (m + 0.5)
    u = np.linspace(-0.5, 0.5, m) if m > 1 else np.zeros(1)
    phi = math.pi + span * u
    kap = kap_mid + (kap_end - kap_mid) * (2 * u) ** 2
    v = np.exp(1j * slope * (phi - math.pi))
    p = np.concatenate([phi, kap, v.real, v.imag])
    if jitter and rng is not None:
        p = p + jitter * rng.standard_normal(p.shape)
    return p


def if_init(m, nu):
    # interference-free-like start: phases pi*l/nu, no DC, equal real v
    phi = math.pi * np.arange(m) / nu
    return np.concatenate([phi, np.zeros(m), np.ones(m), np.zeros(m)])


def resample(prev, m_old, m_new):
    phi, kap = prev[:m_old], prev[m_old:2 * m_old]
    v = prev[2 * m_old:3 * m_old] + 1j * prev[3 * m_old:4 * m_old]
    order = np.argsort(phi); phi, kap, v = phi[order], kap[order], v[order]
    t_old = np.linspace(0, 1, m_old); t_new = np.linspace(0, 1, m_new)
    phi2 = np.interp(t_new, t_old, phi); kap2 = np.interp(t_new, t_old, kap)
    lv = np.interp(t_new, t_old, np.log(np.abs(v) + 1e-300))
    av = np.interp(t_new, t_old, np.unwrap(np.angle(v)))
    v2 = np.exp(lv + 1j * av)
    return np.concatenate([phi2, kap2, v2.real, v2.imag])
