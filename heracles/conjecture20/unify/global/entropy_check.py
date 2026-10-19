# Checks of the temperature/entropy identities.
#  (1) for a fixed network, s = nu/b:  d/ds log gamma(theta;s) = -H(w),  d^2/ds^2 = -Var_w(log w)/s,
#      w_i = |omega_i|^b / sum_j |omega_j|^b  (cost shares).
#  (2) programme: d/ds log Gamma_s(p) = -H(z*),  z_j* = (y_j*)^{1/s} budget shares (away from transitions).
#  (3) IF optimum: H(w) = log nu + H(z*).
#  (4) certified p=3 counterexamples: cost-share entropy vs log nu, and the b where they cross the IF value.
import json, math, numpy as np
from lib import *
from transitions_fast import table

def gam_s(net, s):
    """gamma(theta; s) for the same neurons at b = nu/s."""
    b = net.nu / s
    S = (net.U ** 2).sum((1, 2)) + (net.W ** 2).sum(1)
    return net.margin() / np.sum(S ** (b / 2)) ** s

def H_V(net, s):
    b = net.nu / s
    S = (net.U ** 2).sum((1, 2)) + (net.W ** 2).sum(1)
    w = S ** (b / 2); w = w / w.sum(); w = w[w > 0]
    H = -(w * np.log(w)).sum(); V = (w * np.log(w) ** 2).sum() - H ** 2
    return H, V

rng = np.random.default_rng(3)
print("(1) fixed networks: finite differences in s")
for (p, n, m) in [(5, 2, 4), (7, 3, 5)]:
    U = rng.normal(size=(m, n, p)); W = rng.normal(size=(m, p)); net = Net(U, W, 2.0)
    # make the margin positive: add a strong IF network
    G, y, _ = programme(p, (n + 1) / 2.0)
    IF = if_network(p, n, 3 * y, 2.0)
    net = Net(np.concatenate([IF.U, 0.3 * U]), np.concatenate([IF.W, 0.3 * W]), 2.0)
    for s in [1.2, 1.4]:
        e = 1e-4
        f0, fp, fm = [math.log(gam_s(net, t)) for t in (s, s + e, s - e)]
        H, V = H_V(net, s)
        print(f"  p={p} n={n} s={s}: dlog/ds fd {(fp-fm)/(2*e):.7f}  -H {-H:.7f} | d2 fd {(fp-2*f0+fm)/e**2:.5f}  -V/s {-V/s:.5f}")

print("(2) programme: d/ds log Gamma_s vs -H(z*)  and (3) IF network entropy")
for p, s in [(5, 1.5), (7, 1.3), (7, 2.0), (11, 1.42), (11, 1.8)]:
    e = 1e-4
    lg = [math.log(max(table(p, t).values())) for t in (s - e, s + e)]
    G, y, _ = programme(p, s, restarts=20)
    z = y ** (1 / s); z = z[z > 1e-12]; z = z / z.sum(); Hz = -(z * np.log(z)).sum()
    n = 2 if s <= 1.5 else 3; nu = n + 1
    netIF = if_network(p, n, y, nu / s)
    H, V = H_V(netIF, s)
    print(f"  p={p} s={s}: fd {(lg[1]-lg[0])/(2*e):.6f}  -H(z*) {-Hz:.6f} | IF net H {H:.6f} = log nu + H(z*) {math.log(nu)+Hz:.6f}")

print("(4) certified p=3 networks (b = 2): cost-share entropy and crossing with the IF value")
def p3net(path):
    d = json.load(open(path)); p, n = d['p'], d['n']
    lam, lv, av = map(np.array, (d['lam'], d['logV'], d['argV']))
    lv = lv - lv.max()
    eps = np.array([1.0 if a < p / 2 else -1.0 for a in range(p)])
    ea = np.array([2 * a if a < p / 2 else 2 * a - p for a in range(p)], float)
    rho = np.exp(2j * np.pi * np.arange(p) / p)
    U, W = [], []
    for i in range(len(lam)):
        u = eps * np.exp(lam[i] * ea); w = np.real(np.exp(lv[i] + 1j * av[i]) * rho)
        t = (np.linalg.norm(w) * np.linalg.norm(u) ** n) ** (1 / (n + 1))
        U.append(u / np.linalg.norm(u) * t); W.append(w / np.linalg.norm(w) * t)
    return p, n, np.array(U), np.array(W)

def counts(n, p):
    def rec(prefix, left, k):
        if k == p - 1:
            yield prefix + (left,); return
        for v in range(left + 1):
            yield from rec(prefix + (v,), left - v, k + 1)
    yield from rec((), n, 0)

base = "/home/user/grok/heracles/conjecture20/oned_certified/"
for n in range(11, 19):
    p, n, U, W = p3net(base + f"p3_n{n}.json"); nu = n + 1
    h = np.inf
    for N in counts(n, p):
        N = np.array(N); y = int((np.arange(p) * N).sum() % p)
        z = np.prod(U ** N[None, :], axis=1); f = z @ W
        h = min(h, min(f[y] - f[c] for c in range(p) if c != y))
    S = nu * (U ** 2).sum(1)                                     # |omega_i|^2 (balanced, slot-symmetric)
    kmin = 1 - math.cos(2 * math.pi / 3)
    def ratio(b):
        s = nu / b
        return h / np.sum(S ** (b / 2)) ** s / (nu ** (1 - s) * Bp(3, n) * kmin)
    w = S / S.sum(); H = -(w * np.log(w)).sum()
    # crossing: largest b in [2, nu] with ratio >= 1
    bs = np.linspace(2, 3, 2001); rs = np.array([ratio(b) for b in bs])
    cross = bs[np.where(rs >= 1)[0].max()] if (rs >= 1).any() else None
    print(f"  n={n} nu={nu} width={len(U)}: ratio(b=2) {ratio(2.0):.5f}  H(w) {H:.4f}  log(width) {math.log(len(U)):.4f}  log nu {math.log(nu):.4f}  ratio>=1 up to b={cross}")
