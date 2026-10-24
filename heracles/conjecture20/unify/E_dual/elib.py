# E_dual helper library: Renyi/entropy functionals of networks.  Network machinery is imported read-only
# from unify/global/lib.py (Net, if_network, programme, ...).
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'global'))
import lib as G  # noqa: E402


def nu_costs(net):
    """a_i = ||omega_i||^nu (s-independent)."""
    S = (net.U ** 2).sum((1, 2)) + (net.W ** 2).sum(1)
    return S ** (net.nu / 2.0)


def shares(a, s):
    """cost shares at temperature s: w_i = a_i^{1/s} / sum a^{1/s}."""
    x = a ** (1.0 / s)
    return x / x.sum()


def shannon(w):
    w = w[w > 0]
    return float(-(w * np.log(w)).sum())


def renyi(p, alpha):
    p = p[p > 0]
    if abs(alpha - 1) < 1e-12:
        return shannon(p)
    return float(np.log((p ** alpha).sum()) / (1 - alpha))


def kl(w, p):
    m = w > 0
    return float((w[m] * np.log(w[m] / p[m])).sum())


def gamma_s(net, s):
    """log gamma(theta;s) = log h - s log sum a_i^{1/s} for the network's table (independent of net.b)."""
    a = nu_costs(net)
    h = net.margin()
    return h, np.log(h) - s * np.log((a ** (1.0 / s)).sum()) if h > 0 else -np.inf


def random_net(p, n, m, rng, dc=True, b=2.0):
    U = rng.standard_normal((m, n, p)); W = rng.standard_normal((m, p))
    if not dc:
        U -= U.mean(2, keepdims=True); W -= W.mean(1, keepdims=True)
    return G.Net(U, W, b)
