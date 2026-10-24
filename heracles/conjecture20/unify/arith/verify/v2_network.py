# Referee: EXACT network-level check of the relations-case chain (R1, R2, R3, R5, R6) on genuine class-C n=2 networks.
# Networks are built neuron by neuron (u_1,u_2,w single-frequency cosines), the full table f[x1,x2,c] is formed, and every
# quantity (h, cost, atoms, sign-flipped coefficients, fields) is read off the TABLE, not from a model.
import numpy as np, sys
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/arith2')
import lib
from scipy.optimize import minimize

def build(p, feats):
    """feats: list of (l, list of atoms (m, a, b, c)) ; returns list of neurons (l, alpha, phi1, phi2, psi)."""
    neurons = []
    for l, atoms in feats:
        for (m, a, b, c) in atoms:
            alpha = (4*m)**(1/3)
            neurons.append((l, alpha, (a-b)/2, (a-c)/2, -(b+c)/2))
    return neurons

def table(p, neurons):
    x = np.arange(p); f = np.zeros((p, p, p))
    for (l, al, p1, p2, ps) in neurons:
        u1 = al*np.cos(2*np.pi*l*x/p+p1); u2 = al*np.cos(2*np.pi*l*x/p+p2); w = al*np.cos(2*np.pi*l*x/p+ps)
        f += u1[:, None, None]*u2[None, :, None]*w[None, None, :]
    return f

def analyse(p, s, neurons, Gam, b=None):
    r = 1/s; nu = 3; bexp = b if b is not None else 3/s
    c = 1-(2/3)**(s-1); tp = 1.5**(s-1); F = (p-1)//2; L = np.arange(1, F+1)
    f = table(p, neurons)
    x = np.arange(p); X1, X2 = np.meshgrid(x, x, indexing='ij'); Y = (X1+X2) % p
    fy = np.take_along_axis(f, Y[:, :, None], 2)[:, :, 0]
    gmin = {}
    for d in range(1, p):
        fyd = np.take_along_axis(f, ((Y-d) % p)[:, :, None], 2)[:, :, 0]
        gmin[d] = (fy-fyd).min()
    h = min(gmin.values())
    cost = sum((3*al**2*p/2)**(bexp/2) for (_, al, _, _, _) in neurons)
    normnu = cost**(nu/bexp)
    # Fourier coefficients from the table
    rho = np.exp(2j*np.pi/p)
    def fhat(k1, k2, kc):
        e1 = rho**(-k1*x); e2 = rho**(-k2*x); e3 = rho**(-kc*x)
        return np.einsum('abc,a,b,c->', f, e1, e2, e3)
    beta = np.array([2*p**-3*fhat(l, l, -l) for l in L]); b1 = np.array([2*p**-3*fhat(-l, l, -l) for l in L])
    b2 = np.array([2*p**-3*fhat(l, -l, -l) for l in L]); b12 = np.array([2*p**-3*fhat(-l, -l, -l) for l in L])
    # atoms per neuron (from the neuron's own DFTs)
    Q = np.zeros(F); Lm = np.zeros(F)
    for (l, al, p1, p2, ps) in neurons:
        u1 = al*np.cos(2*np.pi*l*x/p+p1); u2 = al*np.cos(2*np.pi*l*x/p+p2); w = al*np.cos(2*np.pi*l*x/p+ps)
        bi = 2*p**-3*np.conj((w*rho**(-l*x)).sum())*(u1*rho**(-l*x)).sum()*(u2*rho**(-l*x)).sum()
        Q[l-1] += abs(bi)**r; Lm[l-1] += abs(bi)
    a = 3**(1-s)*Q**s
    A = beta.real; v = np.abs(b12)
    # fields from the table: Phi_l(x) = (2/p) rho^{l y} sum_c f[x,c] rho^{-lc}
    fc = np.fft.fft(f, axis=2)   # sum_c f rho^{-kc}
    W = np.zeros(F, complex); fielderr = 0
    for l in L:
        Phi = (2/p)*rho**(l*Y)*fc[:, :, l % p]
        Rm = (Phi-np.conj(beta[l-1]))*rho**(-l*(2*Y))   # should be constant W_l
        W[l-1] = Rm.mean(); fielderr = max(fielderr, np.abs(Rm-W[l-1]).max())
    th = np.angle(W)
    K = lib.kmat(p)                      # rows delta in F
    Kfull = 1-np.cos(2*np.pi*np.outer(np.arange(1, p), L)/p)   # rows delta = 1..p-1
    out = {}
    out['relations_BC'] = max(np.abs(b1).max(), np.abs(b2).max())/max(1e-300, np.abs(beta).max())
    out['fielderr'] = fielderr; out['|W|-v'] = np.abs(np.abs(W)-v).max()
    out['R1 v<=L'] = (v-Lm).max(); out['R1 L<=tau a'] = (Lm-tp*a).max(); out['P12 A<=a+cv'] = (A-a-c*v).max()
    wv = np.arange(p)
    phi = (v[:, None]*np.cos(2*np.pi*np.outer(L, wv)/p+th[:, None])).sum(0)
    O1 = []; R2a = []; R2b = []; R3 = []; CP = []
    for di, d in enumerate(range(1, p)):
        kap = Kfull[di]
        D = phi-np.roll(phi, d); osc = 0.5*(D.max()-D.min())
        O1.append((kap*A).sum()-osc)
        par = np.sqrt((v**2*kap).sum())
        R2a.append((kap*a).sum()+c*(kap*v).sum()-par)
        R2b.append((kap*a).sum()*(1+tp*max(c-np.sqrt(2/p), 0)))
        R3.append((kap*a).sum()+tp*lib.waterfill_excess(a, kap, c))
        psi = ((v*kap)[:, None]*np.cos(2*np.pi*np.outer(L, wv)/p+th[:, None])).sum(0)
        CP.append((kap*A).sum()+psi.min())
    rows = A[:, None]+v[:, None]*np.cos(2*np.pi*np.outer(L, wv)/p+th[:, None])
    na = (a[a > 0]**r).sum()**s
    LR = Gam*min((np.maximum(rows[:, w], 0)**r).sum()**s for w in range(p))
    EW = Gam*((np.maximum(rows, 0)**r).sum(0).mean())**s
    Nc = int(np.floor(1/(2*c*c))); T = np.argsort(-a)[:Nc]; aT = tp*a.copy(); aT[T] = a[T]
    m_a = (K@a).min(); e = m_a/(Gam*na)
    Bp = 2*(2*p*3)**(-1.5)
    IF = 3**(1-s)*Bp*Gam
    out['h'] = h
    out['h-O1'] = h-min(O1); out['O1-CP(pointwise)'] = max(o-cp for o, cp in zip(O1, CP)); out['minCP-LR'] = min(CP)-LR
    out['h-R2a'] = h-min(R2a); out['h-R2b'] = h-min(R2b); out['h-R3'] = h-min(R3); out['h-R3ii'] = h-(K@aT).min()
    out['h-EW'] = h-EW; out['EW/(Gam||a||)'] = EW/(Gam*na)
    out['||a||-budget'] = na-3**(1-s)*Bp*normnu
    out['gamma/IF'] = h/normnu/IF; out['e(a)'] = e
    out['h/(Gam||a||)'] = h/(Gam*na)
    return out

def rand_feature(rng, kind, target_a, s):
    """atoms (m,a,b,c) with B=C=0, scaled so that the IF price 3^{1-s} Q^s equals target_a."""
    r = 1/s; g = rng.uniform(0, 2*np.pi)
    if kind == 'IF':
        b = 2*np.pi*np.arange(3)/3+rng.uniform(0, 2*np.pi); atoms = [(1.0, 0.0, bi, bi+g) for bi in b]
    elif kind == 'two':
        b0 = rng.uniform(0, 2*np.pi); atoms = [(1.0, 0.0, b0, b0+g), (1.0, 0.0, b0+np.pi, b0+np.pi+g)]
    else:
        k = rng.integers(2, 6); z = rng.normal(size=k)+1j*rng.normal(size=k); z -= z.mean()
        sg = rng.choice([-1, 1]); aph = rng.normal(scale=rng.uniform(0, 1.5), size=k)
        atoms = [(abs(zi), aph[i], np.angle(zi), sg*np.angle(zi)+g) for i, zi in enumerate(z)]
    Q = sum(m**r for (m, _, _, _) in atoms); scale = (target_a/(3**(1-s)*Q**s))
    return [(m*scale, a_, b_, c_) for (m, a_, b_, c_) in atoms]

if __name__ == '__main__':
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    worst = {}
    for p in [5, 7, 11, 13, 17, 23]:
        for s in [1.5, 1.25, 1.1]:
            Gam, y = lib.prog(p, s, restarts=4)
            for trial in range(6):
                feats = []
                for l in range(1, (p-1)//2+1):
                    ta = y[l-1] if trial % 2 == 0 else y[l-1]*rng.uniform(0.3, 1.5)+(0.05*y.max() if rng.random() < 0.3 else 0)
                    if ta <= 1e-12: continue
                    kind = rng.choice(['IF', 'two', 'mix'], p=[0.3, 0.4, 0.3])
                    feats.append((l, rand_feature(rng, kind, ta, s)))
                o = analyse(p, s, build(p, feats), Gam)
                for k, val in o.items():
                    worst[k] = max(worst.get(k, -np.inf), val)
    for k, val in worst.items():
        print(f"max {k:20s} = {val:+.3e}")
