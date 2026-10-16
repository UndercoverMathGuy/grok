# Independent high-precision certifier for slot-symmetric product networks on modular addition, p = 3, b = 2.
# Input: family parameters (phi_i, kappa_i, v_i) from fam.py.  The parameters are first ROUNDED to D decimal
# digits (so the certified network has short explicit decimal weights), then the real network
#     u_{it}(a) = lam_i * g_i(a)/||g_i||   for every slot t,   g_i(a) = kappa_i + 2 cos(2 pi a/3 + phi_i)
#     w_i(c)    = (2/3) Re( W_i' w^{-c} ),  w = e^{2 pi i/3}
# is built and its logits f(x)[c] = sum_i w_i(c) prod_t u_{it}(x_t) are evaluated directly (no Fourier/cone
# formulas).  Because every neuron uses the same factor in every slot, f(x) depends on x only through the counts
# N_a = #{t : x_t = a}, so min over count vectors = min over all 3^n inputs.
import sys, itertools, json
import numpy as np
from mpmath import mp, mpf, mpc, cos, sin, pi, sqrt, exp, nstr, log

mp.dps = 60


def build(params, n, m, digits):
    P = [mpf(str(round(float(x), digits))) for x in params]
    phi, kap = P[:m], P[m:2 * m]
    v = [mpc(P[2 * m + i], P[3 * m + i]) for i in range(m)]
    nu = n + 1
    w = exp(2j * pi / 3)
    U, Wout = [], []
    for i in range(m):
        g = [kap[i] + 2 * cos(2 * pi * a / 3 + phi[i]) for a in range(3)]
        gn = sqrt(sum(x * x for x in g))
        Wi = v[i] * exp(-1j * n * phi[i])                    # network with u = g (unnormalised)
        lam = (sqrt(mpf(2) / 3) * abs(Wi) * gn ** n) ** (mpf(1) / nu)
        u = [lam * x / gn for x in g]
        Wp = Wi * (gn / lam) ** n
        wc = [(mpf(2) / 3) * (Wp * w ** (-c)).real for c in range(3)]
        U.append(u); Wout.append(wc)
    return U, Wout


def certify(params, n, m, digits=12):
    U, Wout = build(params, n, m, digits)
    nu = n + 1
    # cost P_2 = sum_i ( n ||u_i||^2 + ||w_i||^2 )
    P2 = sum(n * sum(x * x for x in U[i]) + sum(x * x for x in Wout[i]) for i in range(m))
    hmin = None
    for N0 in range(n + 1):
        for N1 in range(n + 1 - N0):
            N2 = n - N0 - N1
            y = (N1 + 2 * N2) % 3
            z = [U[i][0] ** N0 * U[i][1] ** N1 * U[i][2] ** N2 for i in range(m)]
            f = [sum(Wout[i][c] * z[i] for i in range(m)) for c in range(3)]
            gap = min(f[y] - f[c] for c in range(3) if c != y)
            hmin = gap if hmin is None else min(hmin, gap)
    gamma = hmin / P2 ** (mpf(nu) / 2)
    gIF = 3 * nu * (6 * mpf(nu) ** 2) ** (-mpf(nu) / 2)        # Theorem 8 value at p=3, b=2
    return hmin, P2, gamma, gIF, gamma / gIF, U, Wout


if __name__ == "__main__":
    f, n, m = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    digits = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    params = np.load(f)
    h, P2, gam, gIF, rat, U, W = certify(params, n, m, digits)
    print("n=%d m=%d (nu=%d)  parameters rounded to %d digits" % (n, m, n + 1, digits))
    print("  min gap h          =", nstr(h, 25))
    print("  P_2 = sum ||w||^2  =", nstr(P2, 25))
    print("  normalized margin  =", nstr(gam, 25))
    print("  interference-free  =", nstr(gIF, 25))
    print("  RATIO              =", nstr(rat, 25))
