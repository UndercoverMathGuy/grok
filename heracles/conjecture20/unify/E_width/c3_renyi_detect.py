# (R) Renyi form of the temperature identity: gamma(theta;s) = gamma(theta;1) exp(-(s-1) H_{1/s}(m_theta)),
#     and int_1^s H(theta,sigma) dsigma = (s-1) H_{1/s}(m_theta), m_i = ||omega_i||^nu / sum.
# (D) delta-Fourier isolation for EVERY non-clean character chi != 0 and output j:
#     sum_delta rho^{j delta} E_x[g rho^{-<chi,x>}] = -(p/2) phi_j(chi);  and the convex-end bound
#     h <= (p/(p-1)) [ sum_l Re beta_l - |phi_j(chi)|/2 ]  =>  gamma(theta;1) <= B_p Gamma_1 (1 - M/(2 B_p P_nu)).
# (X) split exchange identity at b = 2 (no DC): (nu/2-1)[H_{2/nu}(m_split) - H_{2/nu}(m)] = log P_nu(theta) - log P_nu(split).
# (O) one neuron has h <= 0 for every n >= 2, p >= 3.
import sys, numpy as np, itertools
from elib import *
from scipy.integrate import quad
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/global')
import lib as glib

rng = np.random.default_rng(3)
# ---------- (R)
errR = 0; errI = 0
for trial in range(200):
    m = rng.integers(1, 12); a = rng.pareto(1.0, m) + 1e-3
    h = 1.0; nu = 4
    for s in [1.3, 2.0]:
        r = 1 / s
        lg = np.log(h) - s * np.log(np.sum(a ** r)); lg1 = np.log(h) - np.log(a.sum())
        m_ = a / a.sum()
        errR = max(errR, abs(lg - (lg1 - (s - 1) * renyi(m_, r))))
        Hs = lambda sig: shannon(a ** (1 / sig))
        errI = max(errI, abs(quad(Hs, 1, s, epsabs=1e-12)[0] - (s - 1) * renyi(m_, r)))
print(f"(R) Renyi identity max error {errR:.1e}; integral of cost-share entropy = (s-1) H_(1/s): max error {errI:.1e}")

# ---------- (D)
worstD = 0; worstB = -np.inf; cnt = 0
for (p, n, m) in [(5, 2, 5), (7, 2, 4), (5, 3, 4), (3, 3, 4), (11, 2, 5), (7, 3, 3)]:
    F = (p - 1) // 2
    for trial in range(15):
        if trial % 3 == 0:
            G, ys = programme_vertices(p, 1.0); U, W = if_network(p, n, ys[0])
            U = np.concatenate([U + 0.05 * rng.normal(size=U.shape), 0.3 * rng.normal(size=(2, n, p))])
            W = np.concatenate([W + 0.05 * rng.normal(size=W.shape), 0.3 * rng.normal(size=(2, p))])
            net = Net(U, W)
        else:
            net = Net(rng.normal(size=(m, n, p)), rng.normal(size=(m, p)))
        g = net.gaps(); h = g.min(); phi = net.fields(); beta = net.clean()
        y = net.Y; X = net.X
        for (j, chi) in [(rng.integers(1, F + 1), tuple(rng.integers(0, p, n))) for _ in range(6)]:
            if not any(chi): continue
            ph = np.exp(-2j * np.pi * (X @ np.array(chi)) / p)
            lhs = sum(np.exp(2j * np.pi * j * d / p) * np.mean(g[:, d - 1] * ph) for d in range(1, p))
            worstD = max(worstD, abs(lhs + (p / 2) * phi[j - 1][chi]) / (1 + abs(phi[j - 1][chi])))
        M = max(abs(phi[j - 1][chi]) for j in range(1, F + 1) for chi in itertools.product(range(p), repeat=n) if any(chi))
        bound = (p / (p - 1)) * (np.sum(beta.real) - M / 2)
        worstB = max(worstB, h - bound); cnt += 1
        P_nu = np.sum(net.mass())
        assert np.sum(np.abs(beta)) <= Bp(p, n) * P_nu * (1 + 1e-12)
print(f"(D) identity sum_delta rho^(j delta) E[g rho^-<chi,x>] = -(p/2) phi_j(chi): max rel error {worstD:.1e}")
print(f"(D) convex-end bound h - (p/(p-1))[sum Re beta - M/2]: max over {cnt} networks = {worstB:.3e} (must be <= 0)")

# ---------- (X)
worstX = 0
for (p, n, m) in [(5, 2, 4), (7, 3, 3), (11, 2, 3), (5, 4, 3)]:
    for trial in range(10):
        U = rng.normal(size=(m, n, p)); W = rng.normal(size=(m, p))
        U -= U.mean(2, keepdims=True); W -= W.mean(1, keepdims=True)       # remove DC
        nu = n + 1; F = (p - 1) // 2
        def proj(v, k):
            vh = np.fft.fft(v, axis=-1); mask = np.zeros(p); mask[k] = 1; mask[(-k) % p] = 1
            return np.real(np.fft.ifft(vh * mask, axis=-1))
        Us = np.concatenate([proj(U, k) for k in range(1, F + 1)]); Ws = np.concatenate([proj(W, k) for k in range(1, F + 1)])
        th = Net(U, W); sp = Net(Us, Ws)
        assert abs(th.cost(2) - sp.cost(2)) < 1e-9 * th.cost(2)
        lhs = (nu / 2 - 1) * (renyi(sp.mass(), 2 / nu) - renyi(th.mass(), 2 / nu))
        rhs = np.log(np.sum(th.mass())) - np.log(np.sum(sp.mass()))
        worstX = max(worstX, abs(lhs - rhs))
        assert np.allclose(th.clean(), sp.clean())
print(f"(X) split exchange identity at b=2: max error {worstX:.1e}; P_2 equal, clean atoms equal")

# ---------- (O)
best = -np.inf
for (p, n) in [(3, 2), (5, 2), (3, 4), (5, 3), (7, 2), (3, 5), (5, 4)]:
    for seed in range(4):
        r0 = np.random.default_rng(100 + seed)
        U0 = r0.normal(size=(1, n, p)); W0 = r0.normal(size=(1, p))
        net = glib.optimise(U0, W0, 2.0, taus=(3e-2, 1e-2, 3e-3, 1e-3), iters=800)
        best = max(best, net.margin())
    for t in range(2000):
        nn = Net(rng.normal(size=(1, n, p)), rng.normal(size=(1, p)))
        best = max(best, nn.margin() / nn.cost(2) ** ((n + 1) / 2))
print(f"(O) single neuron: max normalised margin found over random + optimised single neurons = {best:.3e} (theorem: <= 0)")
