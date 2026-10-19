# Sanity checks of hlib: class-C model vs refereed full-table library; gradients; IF value; mpmath margin.
import numpy as np, time
import hlib as H
GL = H.GL
rng = np.random.default_rng(1)
for (p, n) in [(5, 2), (7, 2), (11, 2), (5, 3), (7, 3), (5, 4)]:
    F = (p - 1) // 2
    freqs = list(rng.integers(1, F + 1, size=6))
    for b in [2.0, 2.5]:
        s = (n + 1) / b
        cc = H.CC(p, n, freqs, s)
        x = rng.standard_normal(2 * cc.m + cc.m * n)
        U, W = cc.to_net(x)
        net = GL.Net(U, W, b)
        g_full = net.gaps(); g_mod = cc.gaps(x)
        # cost: full cost vs B_p^{-r} sum |b|^r
        bb, _ = cc.unpack(x)
        c_mod = H.Bp(p, n) ** (-1 / s) * np.sum(np.abs(bb) ** (1 / s))
        # clean coefficient check: conj(beta_l) = sum b_i
        cl = net.clean()
        Asum = np.array([bb[np.array(freqs) == l].sum() for l in range(1, F + 1)])
        print(f"p={p} n={n} b={b}: max|g_full-g_model|={np.abs(g_full - g_mod).max():.2e} "
              f"cost rel err={abs(net.cost() - c_mod) / c_mod:.2e} clean err={np.abs(np.conj(cl) - Asum).max():.2e}")
# gradient check
p, n, s = 7, 3, 4 / 2.0
cc = H.CC(p, n, [1, 1, 2, 3, 3], s)
x = rng.standard_normal(2 * cc.m + cc.m * n)
tau = 0.05
F0, g0 = cc.soft(x, tau)
num = np.zeros_like(x); e = 1e-6
for k in range(x.size):
    xp = x.copy(); xp[k] += e; xm = x.copy(); xm[k] -= e
    num[k] = (cc.soft(xp, tau)[0] - cc.soft(xm, tau)[0]) / (2 * e)
print("grad check rel err", np.abs(num - g0).max() / np.abs(num).max())
# jacobian check
q = cc.gaps(x).ravel() / cc.cost(x) ** cc.s
idx = np.argsort(q)[:7]
qk, J = cc.jac_active(x, idx)
Jn = np.zeros_like(J)
for k in range(x.size):
    xp = x.copy(); xp[k] += e; xm = x.copy(); xm[k] -= e
    Jn[:, k] = ((cc.gaps(xp).ravel() / cc.cost(xp) ** cc.s)[idx] - (cc.gaps(xm).ravel() / cc.cost(xm) ** cc.s)[idx]) / (2 * e)
print("jac check rel err", np.abs(Jn - J).max() / np.abs(Jn).max(), "q err", np.abs(qk - q[idx]).max())
# IF value at programme optimum
for (p, n, b) in [(11, 2, 2.0), (13, 2, 2.5), (7, 3, 2.0), (5, 4, 2.0)]:
    s = (n + 1) / b
    G, y, bind = H.programme(p, s)
    Ge, ye = H.programme_exact(p, s)
    S = [l for l in range(1, (p - 1) // 2 + 1) if y[l - 1] > 0]
    freqs = [l for l in S for _ in range(n + 1)]
    cc = H.CC(p, n, freqs, s)
    x = H.if_seed(cc, y)
    ratio = cc.gnorm(x) / ((n + 1) ** (1 - s) * G)
    t = time.time(); x2 = cc.lbfgs(H.if_seed(cc, y, 0.05, rng)); t1 = time.time() - t
    x3, v3 = cc.slp(x2)
    print(f"IF p={p} n={n} b={b}: Gamma={G:.10f} exact={Ge:.10f} S={S} bind={bind} ratio={ratio:.12f} "
          f"jitter+lbfgs={cc.gnorm(x2)/((n+1)**(1-s)*G):.9f} ({t1:.1f}s) +slp={v3/((n+1)**(1-s)*G):.12f}")
# mpmath margin of a small network
cc = H.CC(5, 2, [1, 1, 1, 2, 2, 2], 1.5)
G, y, _ = H.programme(5, 1.5)
x = H.if_seed(cc, y)
U, W = cc.to_net(x)
hm, ct = H.margin_mp(U, W)
print("mp margin", float(hm), "float model margin", cc.gaps(x).min() * H.Bp(5, 2) * 0 + GL.Net(U, W, 2.0).margin())
