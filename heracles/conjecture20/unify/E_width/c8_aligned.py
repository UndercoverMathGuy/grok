# Theorem W8 (cancellation-free circular single features are never cheap), checks:
#  class-C frequency-1 single-feature networks with phase-aligned atoms (arg beta_i all equal):
#  (a) cone L2:  sum_{T != 0} |beta_T|^2 <= Re((A-c)^2)/cos(2pi/p) - |A-c|^2,  c = min_x cm(Phi(x))
#  (b) chain:    gamma/gamma_IF <= (c/L) (1 + (2/(nu L^2)) sum_{intervals I} |beta_I|^2)^{s-1} <= (1-u) e^{k_p u^2} <= 1
#  plus an optimisation of gamma/gamma_IF over aligned networks (numerical evidence of sharpness / no counterexample)
import numpy as np, itertools
from scipy.optimize import minimize
from elib import *

rng = np.random.default_rng(8)


def build(p, n, amp, ph_in, j=1):
    """class-C frequency-j neurons; output phase chosen so that every atom beta_i is real positive."""
    m = len(amp); a = np.arange(p)
    U = np.zeros((m, n, p)); W = np.zeros((m, p))
    for i in range(m):
        for t in range(n):
            U[i, t] = amp[i] * np.cos(2 * np.pi * j * a / p + ph_in[i, t])
        # beta_i ~ conj(what(j)) prod_t uhat_t(j); uhat_t(j) ~ e^{i ph_in}; what(j) ~ e^{i psi} -> beta ~ e^{i(sum ph - psi)}
        psi = ph_in[i].sum()
        W[i] = amp[i] * np.cos(2 * np.pi * j * a / p + psi)
    return U, W


def analyse(net, p, n, s):
    F = (p - 1) // 2; nu = n + 1
    phi = net.fields()[0]                                 # feature 1 field coefficients
    A = phi[(0,) * n]
    # field values Phi(x) = sum_chi phi(chi) rho^{<chi,x>}
    Phi = np.fft.ifftn(phi) * p ** n
    th = np.pi / p
    cm = Phi.real - np.abs(Phi.imag) / np.tan(th)
    c = cm.min()
    sumT = sum(abs(phi[tuple((2 * T[t]) % p for t in range(n))]) ** 2 for T in itertools.product([0, 1], repeat=n) if any(T))
    beta, xi = net.atoms(); b1 = beta[:, 0]; x1 = xi[:, :, 0]
    L = np.abs(b1).sum()
    intervals = [(a, b) for a in range(n) for b in range(a + 1, n + 1)]
    sumI = sum(abs(np.sum(b1 * np.prod(x1[:, a:b], axis=1))) ** 2 for (a, b) in intervals)
    kmin = 1 - np.cos(2 * np.pi / p)
    g = net.gamma(s); gif = nu ** (1 - s) * Bp(p, n) * kmin
    coneL2 = (((A - c) ** 2).real / np.cos(2 * th) - abs(A - c) ** 2) - sumT
    bound = (c / L) * (1 + 2 * sumI / (nu * L ** 2)) ** (s - 1)
    kp = 1 / np.cos(2 * th) - 1
    return g / gif, bound, coneL2, c, abs(A), L, kp


worst = {'cone': np.inf, 'chain': -np.inf, 'final': -np.inf}
tested = 0
for (p, n) in [(5, 2), (7, 2), (5, 3), (7, 3), (11, 2), (5, 4)]:
    nu = n + 1
    for trial in range(40):
        m = nu + rng.integers(0, 4)
        base = np.array([np.pi * (l % nu) / nu for l in range(m)])
        ph = np.repeat(base[:, None], n, axis=1) + rng.normal(scale=[0.0, 0.05, 0.2, 0.5][trial % 4], size=(m, n))
        amp = np.where(np.arange(m) < nu, 1.0, 0.3) * np.exp(rng.normal(scale=0.2, size=m))
        U, W = build(p, n, amp, ph)
        net = Net(U, W)
        if net.margin() <= 0: continue
        beta, _ = net.atoms(); assert np.max(np.abs(np.angle(beta[:, 0]))) < 1e-9   # aligned
        for s in [1.0, 1.3, nu / 2]:
            ratio, bound, coneL2, c, A, L, kp = analyse(net, p, n, s)
            worst['cone'] = min(worst['cone'], coneL2)
            worst['chain'] = max(worst['chain'], ratio - bound)
            u = 1 - c / L
            worst['final'] = max(worst['final'], ratio - 1)
            tested += 1
print(f"{tested} (network, s) pairs; min cone-L2 slack = {worst['cone']:.3e} (>= 0); max gamma/gIF - bound = {worst['chain']:.3e} (<= 0); max gamma/gIF - 1 = {worst['final']:.3e} (<= 0)")

# optimisation over aligned class-C single features (amplitudes + input phases free; output phase forced)
for (p, n, b, m) in [(5, 2, 2.0, 4), (5, 3, 2.0, 5), (7, 3, 2.0, 5), (5, 4, 2.0, 6), (7, 2, 2.5, 3)]:
    nu = n + 1; s = nu / b
    kmin = 1 - np.cos(2 * np.pi / p); gif = nu ** (1 - s) * Bp(p, n) * kmin
    def negsoft(z, tau):
        amp = np.exp(z[:m]); ph = z[m:].reshape(m, n)
        U, W = build(p, n, amp, ph); net = Net(U, W)
        g = net.gaps().ravel() / net.cost(b) ** s / gif
        return -(g.min() - tau * np.log(np.mean(np.exp(-(g - g.min()) / tau))))
    best = -np.inf
    for seed in range(6):
        r0 = np.random.default_rng(seed)
        z = np.concatenate([r0.normal(scale=0.3, size=m), r0.uniform(0, 2 * np.pi, m * n)])
        for tau in [0.1, 0.03, 0.01, 0.003, 0.001]:
            z = minimize(negsoft, z, args=(tau,), method='L-BFGS-B', options={'maxiter': 800}).x
        amp = np.exp(z[:m]); ph = z[m:].reshape(m, n); net = Net(*build(p, n, amp, ph))
        best = max(best, net.gamma(s) / gif)
    print(f"aligned class-C single features, (p,n,b,m)=({p},{n},{b},{m}): best gamma/gamma_IF found = {best:.6f} (theorem: <= 1)")
print("k_p = sec(2pi/p)-1:", {p: round(1 / np.cos(2 * np.pi / p) - 1, 5) for p in [5, 7, 11, 13]}, " sqrt5 =", round(np.sqrt(5), 5))
u2 = (1 + np.sqrt(1 - 2 / np.sqrt(5))) / 2
print(f"elementary lemma: -log(1-u) - sqrt5 u^2 at the interior local min u2={u2:.6f}: {-np.log(1 - u2) - np.sqrt(5) * u2 ** 2:.6f} (> 0)")
uu = np.linspace(0, 0.999999, 200001)
print(f"   min over [0,1) on a grid of -log(1-u) - sqrt5 u^2 (excluding u=0 neighbourhood u<1e-3): {np.min((-np.log(1 - uu) - np.sqrt(5) * uu ** 2)[uu > 1e-3]):.6f}")
