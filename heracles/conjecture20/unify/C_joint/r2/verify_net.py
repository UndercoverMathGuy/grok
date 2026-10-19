# Independent verification of single-T networks (alpha_l IF block + beta_l n-block with phase phi_l) from raw tables.
import numpy as np, itertools, sys
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/C_joint')
from cjlib import gamma_lp, gamma_exact
def neuron(p, l, n, c, xi):
    a = np.arange(p); th = np.angle(xi)/2
    U = np.array([np.cos(2*np.pi*l*a/p + th[t]) for t in range(n)])
    W = 2**n*abs(c)*np.cos(2*np.pi*l*a/p + np.angle(c) + th.sum())
    return U, W
def build(p, n, al, be, ph):
    F = (p-1)//2; nu = n+1; Us = []; Ws = []; feat = []
    for l in range(1, F+1):
        if al[l-1] > 0:
            for k in range(nu):
                U, W = neuron(p, l, n, al[l-1]/nu, np.array([np.exp(2j*np.pi*k/nu)]*n)); Us.append(U); Ws.append(W); feat.append(l)
        if be[l-1] > 0:
            for k in range(n):
                xi = np.array([np.exp(2j*np.pi*k/n)]*n, dtype=complex); xi[0] *= np.exp(1j*ph[l-1])
                U, W = neuron(p, l, n, be[l-1]/n, xi); Us.append(U); Ws.append(W); feat.append(l)
    return np.array(Us), np.array(Ws), np.array(feat)
def check(p, n, s, al, be, ph, Gam=None):
    r = 1/s; nu = n+1; F = (p-1)//2; rho = np.exp(2j*np.pi/p)
    U, W, feat = build(p, n, al, be, ph); m = len(W)
    X = np.array(list(itertools.product(range(p), repeat=n))); y = X.sum(1) % p
    P = np.ones((m, len(X)))
    for t in range(n): P *= U[:, t, X[:, t]]
    f = P.T @ W
    fc = np.fft.fft(f, axis=1)
    Phi = np.array([(2.0/p)*rho**(l*y)*fc[:, l] for l in range(1, F+1)])
    pred = np.array([al[l-1] + be[l-1]*(1 + np.exp(1j*ph[l-1])*rho**(2*l*y)) for l in range(1, F+1)])
    ferr = np.max(np.abs(Phi - pred))/max(1e-300, np.max(np.abs(pred)))
    Uh = np.fft.fft(U, axis=2); Wh = np.fft.fft(W, axis=1)
    bet = np.array([2*p**(-nu)*np.conj(Wh[i, feat[i]])*np.prod(Uh[i, :, feat[i]]) for i in range(m)])
    Ql = np.zeros(F); np.add.at(Ql, feat-1, np.abs(bet)**r)
    psi = (np.maximum(Phi.real, 0)**r).sum(0)
    pw = psi.min()/(nu**(r-1)*Ql.sum())
    h = np.inf
    for d in range(1, p): h = min(h, (f[np.arange(len(X)), y] - f[np.arange(len(X)), (y-d) % p]).min())
    norms = np.concatenate([np.linalg.norm(U, axis=2), np.linalg.norm(W, axis=1)[:, None]], axis=1)
    om2 = nu*np.prod(norms, axis=1)**(2.0/nu); b = nu/s
    gam = h/np.sum(om2**(b/2))**(nu/b); Bp = 2*(2*p*nu)**(-nu/2.0)
    if Gam is None: Gam = gamma_exact(p, s)[0] if p <= 23 else gamma_lp(p, s, restarts=30)[0]
    return dict(field_err=ferr, PW_ratio=pw, h=h, gamma_over_IF_upper=gam/(nu**(1-s)*Bp*Gam), Gamma_lower=Gam, width=m)
if __name__ == '__main__':
    for fn in sys.argv[1:]:
        p = int(fn.split('_p')[1].split('_')[0]); n = int(fn.split('_n')[1].split('_')[0]); s = float(fn.split('_s')[1][:-4])
        x = np.load(fn); F = (p-1)//2
        al = np.exp(x[:F]); be = np.exp(x[F:2*F]); ph = x[2*F:]
        al = np.where(al < 1e-9*be.max(), 0.0, al)       # drop negligible IF blocks (only reduces cost)
        print(fn, check(p, n, s, al, be, ph), flush=True)
