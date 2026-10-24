"""Class-C optimum at p = 3 (no DC input components), b = 2: is it above the IF value?

For a class-C network at p = 3 every input factor lies in the frequency-1 plane, so (atom form)
    Phi(x) = sum_i b_i prod_t (1 + xi_it z_t),   z_t = rho^{2 x_t} in mu_3,  |xi_it| = 1,
and after balancing gamma/gamma_IF = min_x cm(Phi(x)) / (nu^{1-s} (sum_i |b_i|^r)^s),  cm(z) = Re z - cot(pi/3)|Im z|.
(Single-neuron budget is attained by balanced single-frequency neurons; Gamma_s(3) = kappa_min(3).)
We maximise this ratio over atoms by smoothed-min L-BFGS from many starts: slot-symmetric atoms (xi_it = xi_i,
margin over count vectors) and general atoms (margin over all 3^n inputs, smaller n).
Run: OMP_NUM_THREADS=1 uv run --with numpy --with torch python p3_classC.py n [N_list] [restarts]
"""
import sys, math, itertools, time
import numpy as np, torch
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

n = int(sys.argv[1]); Ns = [int(v) for v in sys.argv[2].split(',')] if len(sys.argv) > 2 else [n + 1]
R = int(sys.argv[3]) if len(sys.argv) > 3 else 20
mode = sys.argv[4] if len(sys.argv) > 4 else 'sym'
b = 2.0; nu = n + 1; s = nu / b; r = 1 / s
cot = 1 / math.tan(math.pi / 3)
w3 = torch.exp(2j * math.pi * torch.arange(3) / 3)

if mode == 'sym':
    Nc = torch.tensor([(a, c, n - a - c) for a in range(n + 1) for c in range(n + 1 - a)], dtype=torch.float64)
else:
    Z = torch.tensor(list(itertools.product(range(3), repeat=n)))           # all inputs, values of x_t
    Zc = w3[(2 * Z) % 3]                                                     # z_t = rho^{2 x_t}


def ratio(params, N, tau=None):
    br, bi = params[:N], params[N:2 * N]
    bcoef = torch.complex(br, bi)
    if mode == 'sym':
        th = params[2 * N:3 * N]
        xi = torch.exp(1j * th)
        # log prod_a (1 + xi w^a)^{N_a}
        fac = torch.log(1 + xi[:, None] * w3[None, :])                       # N x 3 (complex log)
        L = Nc.to(torch.complex128) @ fac.T                                     # counts x N
        Phi = (torch.exp(L) * bcoef[None, :]).sum(1)
    else:
        th = params[2 * N:].reshape(N, n)
        xi = torch.exp(1j * th)                                              # N x n
        prod = torch.ones(Zc.shape[0], N, dtype=torch.complex128)
        for t in range(n):
            prod = prod * (1 + xi[None, :, t] * Zc[:, t:t + 1])
        Phi = prod @ bcoef
    cm = Phi.real - cot * Phi.imag.abs()
    Q = (bcoef.abs() ** r).sum()
    den = nu ** (1 - s) * Q ** s
    if tau is None:
        return cm.min() / den
    return (-tau * torch.logsumexp(-cm / tau, 0)) / den


def opt(x0, N):
    P = torch.tensor(x0, requires_grad=True)
    for tau_rel in [0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3, 3e-4]:
        o = torch.optim.LBFGS([P], lr=1, max_iter=300, line_search_fn='strong_wolfe')
        def closure():
            o.zero_grad()
            with torch.no_grad():
                sc = (torch.complex(P[:N], P[N:2 * N]).abs() ** r).sum() ** s * nu ** (1 - s)
            loss = -ratio(P, N, tau=float(tau_rel * sc))
            loss.backward(); return loss
        try:
            o.step(closure)
        except Exception:
            pass
    with torch.no_grad():
        return float(ratio(P, N)), P.detach().numpy().copy()


rng = np.random.default_rng(1)
t0 = time.time()
best_all = -1e9
for N in Ns:
    best = -1e9
    for rep in range(R):
        if mode == 'sym':
            if rep == 0 and N >= nu:     # IF block (nu atoms, nu-th roots) + extra tiny atoms
                br = np.r_[np.ones(nu) / nu, 1e-3 * rng.normal(size=N - nu)]
                bi = np.r_[np.zeros(nu), 1e-3 * rng.normal(size=N - nu)]
                th = np.r_[2 * np.pi * np.arange(nu) / nu, rng.uniform(0, 2 * np.pi, N - nu)]
            else:
                br = rng.normal(size=N); bi = rng.normal(size=N); th = rng.uniform(0, 2 * np.pi, N)
            x0 = np.r_[br, bi, th]
        else:
            x0 = np.r_[rng.normal(size=N), rng.normal(size=N), rng.uniform(0, 2 * np.pi, N * n)]
        v, x = opt(x0, N)
        best = max(best, v)
    best_all = max(best_all, best)
    print(f"p=3 class C ({mode}) n={n} N={N}: best gamma/gamma_IF = {best:.6f}   ({time.time() - t0:.0f}s)", flush=True)
print(f"p=3 class C ({mode}) n={n}: overall best {best_all:.6f}")
