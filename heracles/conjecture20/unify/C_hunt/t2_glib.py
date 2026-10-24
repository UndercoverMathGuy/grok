# sanity checks for glib: gradients (with penalty), Jacobian, IF block value, mask exactness
import numpy as np
import hlib as H, glib as Gb
GL = H.GL
rng = np.random.default_rng(3)
p, n, b = 7, 2, 2.0
F = 3
allowed = [[{0, 1, 2}, {1, 3}, {1, 2, 3}], [set(range(4))] * 3, [{1}, {1}, {1}], [{0, 2}, {2, 3}, {2}]]
g = Gb.GNet(p, n, b, allowed)
x = rng.standard_normal(g.nx)
tau = 0.02
for mu in [0.0, 3.0]:
    F0, g0 = g.soft(x, tau, pen=(2, 3), mu=mu)
    num = np.zeros_like(x); e = 1e-6
    for k in range(x.size):
        xp = x.copy(); xp[k] += e; xm = x.copy(); xm[k] -= e
        num[k] = (g.soft(xp, tau, (2, 3), mu)[0] - g.soft(xm, tau, (2, 3), mu)[0]) / (2 * e)
    print("mu", mu, "grad rel err", np.abs(num - g0).max() / np.abs(num).max())
nt = g.net(x); q = nt.gaps().ravel() / nt.cost() ** g.s
idx = np.argsort(q)[:9]
qk, J = g.jac_active(x, idx)
Jn = np.zeros_like(J); e = 1e-6
for k in range(x.size):
    xp = x.copy(); xp[k] += e; xm = x.copy(); xm[k] -= e
    a = g.net(xp); bb = g.net(xm)
    Jn[:, k] = ((a.gaps().ravel() / a.cost() ** g.s)[idx] - (bb.gaps().ravel() / bb.cost() ** g.s)[idx]) / (2 * e)
print("jac rel err", np.abs(J - Jn).max() / np.abs(Jn).max(), "q err", np.abs(qk - q[idx]).max())
# mask exactness: neuron 3 has no freq 1 in slot 0 -> beta_{3,1} = 0 ; neuron 0 has no freq 2 in slot 1 -> beta_{0,2}=0
U, W = g.to_UW(x); at = GL.Net(U, W, b).atoms()
print("beta_{0,2}", abs(at[0, 1]), "beta_{3,1}", abs(at[3, 0]), "beta_{0,1}", abs(at[0, 0]))
# IF block single feature at (5,2): gamma / (3^{1-s} B_p kappa_min) = 1
for (p, n, b) in [(5, 2, 2.0), (7, 3, 2.5), (11, 2, 2.0)]:
    s = (n + 1) / b
    U, W = GL.if_block(p, n, 1, 1.0)
    al = [[{1}] * (n + 1) for _ in range(n + 1)]
    g = Gb.GNet(p, n, b, al); x = g.from_UW(U, W)
    kmin = 1 - np.cos(2 * np.pi / p)
    D = (n + 1) ** (1 - s) * H.Bp(p, n) * kmin
    x2 = g.lbfgs(x + 0.05 * rng.standard_normal(x.size)); x3, v = g.slp(x2)
    print(f"IF single feature ({p},{n},{b}): ratio {g.gamma(x)/D:.12f}; jitter+lbfgs {g.gamma(x2)/D:.9f}; +slp {v/D:.12f}")
