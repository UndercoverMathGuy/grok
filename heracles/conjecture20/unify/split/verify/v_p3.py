"""Referee checks against the p = 3 counterexamples (exact_certificate.py weights, n = 11 and 13), exact arithmetic.
Checks that no SPLIT claim implies the IF value at p = 3:
  * the certificates are NOT interference-free (beta_{1,T} != 0), so R7(a) does not apply to them;
  * their splits (DC removed) have negative margin, R1 cost inequality, R8 bound, R2(4) Jensen over Sigma.
Slot symmetry: every neuron uses the same u_i in all n slots, so margins are minima over count vectors (exact Fractions).
Run: uv run --with numpy python v_p3.py
"""
import importlib.util, itertools, math
from fractions import Fraction as Fr
import numpy as np

spec = importlib.util.spec_from_file_location('excert', '/home/user/grok/heracles/conjecture20/exact_certificate.py')
ex = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)
p = 3


def margin_groups(groups, W):
    """groups: list of (count, U) where U[i] is the input vector used by neuron i in those slots; exact min gap."""
    m = len(W)
    def comps(k):
        return [(a, b, k - a - b) for a in range(k + 1) for b in range(k + 1 - a)]
    best = None
    for combo in itertools.product(*[comps(k) for k, _ in groups]):
        y = sum(N1 + 2 * N2 for (_, N1, N2) in combo) % 3
        z = []
        for i in range(m):
            v = Fr(1)
            for (N0, N1, N2), (_, U) in zip(combo, groups):
                v *= U[i][0] ** N0 * U[i][1] ** N1 * U[i][2] ** N2
            z.append(v)
        f = [sum(W[i][c] * z[i] for i in range(m)) for c in range(3)]
        for c in range(3):
            if c != y:
                g = f[y] - f[c]
                best = g if best is None or g < best else best
    return best


for n, net in ex.NETWORKS.items():
    nu = n + 1; s = Fr(nu, 2)
    U = [[Fr(a) for a in row] for row in net['u']]; W = [[Fr(a) for a in row] for row in net['w']]
    m = len(U)
    h = margin_groups([(n, U)], W)
    P = sum(n * sum(a * a for a in U[i]) + sum(a * a for a in W[i]) for i in range(m))
    gIF = Fr(3 * nu) / Fr(6 * nu * nu) ** (nu // 2)
    gam = h / P ** (nu // 2)
    # split = remove DC (at p = 3, P_1 = I - P_0)
    Us = [[a - sum(U[i]) / 3 for a in U[i]] for i in range(m)]; Ws = [[a - sum(W[i]) / 3 for a in W[i]] for i in range(m)]
    hs = margin_groups([(n, Us)], Ws)
    Ps = sum(n * sum(a * a for a in Us[i]) + sum(a * a for a in Ws[i]) for i in range(m))
    dc = P - Ps
    # atoms and sign-flipped coefficients (by symmetry beta_{1,T} depends on |T| only)
    rho = np.exp(2j * np.pi / 3)
    uh = np.array([sum(float(U[i][a]) * rho ** (-a) for a in range(3)) for i in range(m)])
    wh = np.array([sum(float(W[i][a]) * rho ** (-a) for a in range(3)) for i in range(m)])
    at = 2 * 3.0 ** (-nu) * np.conj(wh) * uh ** n
    xi = np.conj(uh) / uh
    bT = [abs((at * xi ** k).sum()) for k in range(1, n + 1)]
    beta = abs(at.sum())
    # R8 with live split costs (b = 2, one frequency: P_1^live = split energy), kappa_1 = 3/2 at p = 3
    B3 = 2 * (6 * nu) ** (-nu / 2)
    r8 = float(h) / (B3 * float(Ps) ** float(s) * 1.5)
    # R2(4): E over Sigma = {+-1}^n of h(sigma theta) vs h(split); by slot symmetry h depends on |S| only
    Ut = [[2 * sum(U[i]) / 3 - a for a in U[i]] for i in range(m)]          # P_0 u - P_1 u
    Eh = Fr(0)
    hk = []
    for k in range(n + 1):
        sgn = (-1) ** k
        Wk = [[sum(W[i]) / 3 + sgn * (a - sum(W[i]) / 3) for a in W[i]] for i in range(m)]
        groups = [g for g in [(k, Ut), (n - k, U)] if g[0] > 0]
        hk.append(margin_groups(groups, Wk))
        Eh += Fr(math.comb(n, k), 2 ** n) * hk[-1]
    print(f'p=3 n={n}: gamma/gamma_IF = {float(gam / gIF):.6f} (exact {gam > gIF}), DC share = {float(dc / P):.4f}')
    print(f'   |beta_1| = {beta:.4e},  max_k |beta_(1,|T|=k)| = {max(bT):.4e}  (min {min(bT):.3e}) -> NOT interference-free: {max(bT) > 1e-6 * beta}')
    print(f'   split: h_split = {float(hs):.6e} (<0: {hs < 0}), gamma_split/gamma_IF = {float(hs / Ps ** (nu // 2) / gIF):.2f}, cost(split) <= cost: {Ps <= P}')
    print(f'   R8 ratio h / (B_3 (P_live)^s kappa) = {r8:.4e} (<= 1: {r8 <= 1})')
    print(f'   R2(4): E_sigma h(sigma theta) = {float(Eh):.4e} <= h(split) = {float(hs):.4e}: {Eh <= hs};  h(|S|=k) for k=0..n: '
          + ', '.join(f'{float(x):.2e}' for x in hk))
