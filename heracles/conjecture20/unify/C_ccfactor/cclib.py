# C_ccfactor library: feature-level / within-feature accounting of the convex-plus-correction identity.
# Reuses the refereed E_width library (general product networks, exact programme by vertex enumeration).
import sys, itertools
import numpy as np
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/E_width')
import elib as E

Bp = E.Bp
kappa = E.kappa
renyi = E.renyi
programme_vertices = E.programme_vertices


def phi_x(net):
    """Phi[l-1, x] = (2/p) rho^{l y(x)} sum_c f(x)[c] rho^{-lc}, flattened x (N = p^n)."""
    p, n = net.p, net.n; F = (p - 1) // 2
    f = net.table().reshape(net.N, p)
    fc = np.fft.fft(f, axis=1)
    y = net.Y
    return np.array([(2.0 / p) * np.exp(2j * np.pi * l * y / p) * fc[:, l] for l in range(1, F + 1)])


def cvec(p, l):
    """c_l(delta) = 1 - rho^{-l delta}, delta = 1..p-1."""
    d = np.arange(1, p)
    return 1 - np.exp(-2j * np.pi * l * d / p)


def feature_gaps(net):
    """g_l(x,delta) = Re(Phi_l(x) c_l(delta)), shape (F, N, p-1). sum_l g_l = gaps."""
    Phi = phi_x(net); p = net.p; F = (p - 1) // 2
    return np.array([np.real(Phi[l - 1][:, None] * cvec(p, l)[None, :]) for l in range(1, F + 1)])


def cm(z, p):
    return np.real(z) - np.abs(np.imag(z)) / np.tan(np.pi / p)


def neuron_freq(net, tol=1e-9):
    """class-C test: returns (is_classC, l(i)) with l(i) the unique frequency carrying neuron i."""
    p = net.p; F = (p - 1) // 2
    Uh = np.abs(np.fft.fft(net.U, axis=2)); Wh = np.abs(np.fft.fft(net.W, axis=1))
    lab = []; ok = True
    for i in range(net.m):
        en = np.array([Wh[i, k] ** 2 + Wh[i, p - k] ** 2 + (Uh[i, :, k] ** 2 + Uh[i, :, p - k] ** 2).sum()
                       for k in range(1, F + 1)])
        dc = Wh[i, 0] ** 2 + (Uh[i, :, 0] ** 2).sum()
        l = int(np.argmax(en)) + 1; lab.append(l)
        if dc > tol * en.max() or np.sort(en)[-2 if F > 1 else -1] > tol * en.max() and F > 1:
            ok = False
    return ok, np.array(lab)


def rand_classC(p, n, widths, rng, scale=None):
    """widths: dict l -> number of frequency-l neurons; random amplitudes/phases (class C)."""
    a = np.arange(p); Us = []; Ws = []
    for l, w in widths.items():
        for k in range(w):
            amp = rng.uniform(0.3, 1.5, n + 1) * (scale[l] if scale else 1.0)
            ph = rng.uniform(0, 2 * np.pi, n + 1)
            Us.append([amp[t] * np.cos(2 * np.pi * l * a / p + ph[t]) for t in range(n)])
            Ws.append(amp[n] * np.cos(2 * np.pi * l * a / p + ph[n]))
    return E.Net(np.array(Us), np.array(Ws))


def Mcov(y, p):
    """M(y) = min_delta sum_l y_l kappa_l(delta) (y indexed l=1..F)."""
    return float((kappa(p) @ np.asarray(y)).min())


def accounting(net, s, pi_val, labels=None):
    """Feature-level / within-feature accounting for a class-C network (labels l(i)).
    Returns dict with all quantities of the factorisation."""
    p, n, nu = net.p, net.n, net.nu; F = (p - 1) // 2; r = 1.0 / s; B = Bp(p, n); G1 = p / (p - 1)
    if labels is None:
        _, labels = neuron_freq(net)
    a = net.mass(); m = a / a.sum()
    h = net.margin(); beta = net.clean()
    A = np.array([a[labels == l].sum() for l in range(1, F + 1)])          # feature nu-masses
    M = A / a.sum()
    K = np.array([np.exp(renyi(a[labels == l], r)) if A[l - 1] > 0 else 1.0 for l in range(1, F + 1)])
    Pl = np.array([(a[labels == l] ** r).sum() for l in range(1, F + 1)])   # feature b-costs
    eps = np.where(A > 0, np.real(beta) / (B * np.where(A > 0, A, 1)), 0.0)   # feature clean efficiency
    Hm = renyi(m, r)
    # exact chain rule
    Z = (M * K ** (s - 1)).sum()                                             # E_M[K_l^{s-1}]
    zhat = M * K ** (s - 1) / Z
    HA = np.log(Z) / (s - 1)                                                 # Arimoto conditional Renyi
    Hz = renyi(zhat[zhat > 0], r)
    e = h / (B * a.sum())
    Mzh = Mcov(zhat, p)
    Pi_feat = Mzh * np.exp(-(s - 1) * Hz)
    Pi_within = e / (Mzh * Z)
    gamma = net.gamma(s)
    # convex-end deficits (IDENTITY.md)
    at, _ = net.atoms()
    sig = 1 - np.real(at).sum(1) / (B * a)
    Esig = (m * sig).sum()
    cleanmin = Mcov(np.real(beta), p)
    dprog = (G1 * np.real(beta).sum() - cleanmin) / (B * G1 * a.sum())
    dint = (cleanmin - h) / (B * G1 * a.sum())
    Esig_l = np.array([(m[labels == l] * sig[labels == l]).sum() / M[l - 1] if M[l - 1] > 0 else 0
                       for l in range(1, F + 1)])
    MPs = Mcov(Pl ** s, p)
    return dict(gamma=gamma, h=h, e=e, Hm=Hm, HA=HA, Hz=Hz, Z=Z, zhat=zhat, M=M, K=K, Pl=Pl, eps=eps,
                Pi_feat=Pi_feat, Pi_within=Pi_within, Esig=Esig, Esig_l=Esig_l, dprog=dprog, dint=dint,
                cleanmin=cleanmin, MPs=MPs, CD=h / (pi_val * B * MPs), B=B)


def profile(net, l=1):
    """per-offset normalised margins of feature l in isolation:
    Psi[k-1] = min_x g_l(x, delta_k) / (kappa_l(delta_k)), indexed by the offset delta = 1..p-1."""
    p = net.p
    g = feature_gaps(net)[l - 1]
    kap = 1 - np.cos(2 * np.pi * l * np.arange(1, p) / p)
    return g.min(0) / kap


def relabel_copy(U, W, l, p, scale=1.0, shift=None):
    """theta^{(l)}: u(a) -> u(l a), w(c) -> w(l c); optional input translation per slot; table scaled by scale."""
    a = np.arange(p)
    idx = (l * a) % p
    U2 = U[:, :, idx].copy(); W2 = W[:, idx].copy()
    if shift is not None:
        for t in range(U.shape[1]):
            U2[:, t, :] = np.roll(U2[:, t, :], shift[t], axis=1)
        W2 = np.roll(W2, int(np.sum(shift)), axis=1)
    nu = U.shape[1] + 1
    c = scale ** (1.0 / nu)
    return U2 * c, W2 * c
