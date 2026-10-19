# Referee check of R10 on general n=2 networks (DC + mixed frequencies), own split code.
import numpy as np
from netlib import table, gaps, coeffs, kap, Bp
rng = np.random.default_rng(51)
def proj(v, k, p):       # projection of v onto the frequency-k plane (k in F) or constants (k = 0)
    a = np.arange(p)
    if k == 0: return np.full(p, v.mean())
    c = np.cos(2*np.pi*k*a/p); s = np.sin(2*np.pi*k*a/p)
    return (v @ c)/(c @ c)*c + (v @ s)/(s @ s)*s
def split(U1, U2, W, p):
    S1, S2, SW, fr = [], [], [], []
    for i in range(len(W)):
        for k in range(1, (p - 1)//2 + 1):
            S1.append(proj(U1[i], k, p)); S2.append(proj(U2[i], k, p)); SW.append(proj(W[i], k, p)); fr.append(k)
    return np.array(S1), np.array(S2), np.array(SW), np.array(fr)
def per_atoms(U1, U2, W, p):   # atom of each neuron at each l
    Fn = (p - 1)//2; a = np.arange(p); r = np.exp(2j*np.pi/p); out = np.zeros((len(W), Fn), complex)
    for l in range(1, Fn + 1):
        u1 = U1 @ r**(-l*a); u2 = U2 @ r**(-l*a); w = W @ r**(l*a); out[:, l-1] = 2*p**-3*w*u1*u2
    return out
for p in [5, 7]:
    for b in [2.0, 2.5]:
        s = 3/b; r = 1/s; th = 3**(1 - s); c = (1 - th)/np.sqrt(3); kp = (p - 1)/(p + 1); Fn = (p - 1)//2
        w = dict(coef=0, atom=0, mineq=np.inf, mbar=np.inf, chain=np.inf, dec=np.inf, cost=-np.inf); ncond = 0
        for t in range(150):
            m = int(rng.integers(2, 7))
            U1 = rng.normal(size=(m, p)); U2 = rng.normal(size=(m, p)); W = rng.normal(size=(m, p))
            if t % 3 == 1: U1 += 2*rng.normal(size=(m, 1)); U2 += 2*rng.normal(size=(m, 1))     # strong input DC
            if t % 3 == 2:   # class-C core with cheap features + small mixed/DC perturbation
                from netlib import random_classC
                C1, C2, CW, _ = random_classC(p, rng, per=(1, 2))
                eps = 10**rng.uniform(-3, -1)
                U1 = np.r_[C1, eps*U1]; U2 = np.r_[C2, eps*U2 + eps]; W = np.r_[CW, eps*W]; m = len(W)
            S1, S2, SW, fr = split(U1, U2, W, p)
            f = table(U1, U2, W); fs = table(S1, S2, SW); g = gaps(f); gs = gaps(fs); gR = g - gs
            co = coeffs(f); cs = coeffs(fs)
            w['coef'] = max(w['coef'], max(abs(co[k] - cs[k]) for k in co))
            at = per_atoms(U1, U2, W, p); ats = per_atoms(S1, S2, SW, p)
            Q = (np.abs(at)**r).sum(0); Qs = (np.abs(ats)**r).sum(0); w['atom'] = max(w['atom'], np.abs(Q - Qs).max())
            cst = sum((np.sum(U1[i]**2) + np.sum(U2[i]**2) + np.sum(W[i]**2))**(b/2) for i in range(m))
            csts = sum((np.sum(S1[i]**2) + np.sum(S2[i]**2) + np.sum(SW[i]**2))**(b/2) for i in range(len(SW)))
            w['cost'] = max(w['cost'], csts - cst)
            h = g[:, :, 1:].min(); a = th*Q**s
            beta = np.array([co[(l, ())] for l in range(1, Fn + 1)])
            R = np.array([np.sqrt(sum(abs(co[(l, T)])**2 for T in [(1,), (2,), (1, 2)])) for l in range(1, Fn + 1)])
            K = np.array([[kap(l, d, p) for l in range(1, Fn + 1)] for d in range(1, p)])
            Wd = K @ a; ds = int(np.argmin(Wd)) + 1; k = K[ds - 1]; cheap = np.abs(beta) > a + 1e-14
            E = (R**2*k).sum()
            for d in (ds, p - ds):
                Ad = gs[:, :, d].mean() - gs[:, :, d].min(); md = g[:, :, d].mean() - h
                w['mineq'] = min(w['mineq'], md - (Ad - gR[:, :, d].max()))
            mbar = 0.5*((g[:, :, ds].mean() - h) + (g[:, :, p - ds].mean() - h))
            rho_ = 0.5*(gR[:, :, ds].max() + gR[:, :, p - ds].max())
            w['mbar'] = min(w['mbar'], mbar - (np.sqrt(kp*E) - rho_))
            rhs = (a*k).sum() + c*np.sqrt(k[cheap].sum())*np.sqrt(E) + rho_ - np.sqrt(kp*E)
            w['chain'] = min(w['chain'], rhs - h)
            if c*np.sqrt(k[cheap].sum())*np.sqrt(E) + rho_ <= np.sqrt(kp*E):
                ncond += 1; w['dec'] = min(w['dec'], Wd.min() - h)
        print(f"R10 p={p} b={b}: coef diff {w['coef']:.1e}, Q diff {w['atom']:.1e}, max cost(split)-cost {w['cost']:.2e}; "
              f"min[m_d - (A_d(split) - max g_R)] {w['mineq']:.3e}; min[mbar - (sqrt(kE) - rho)] {w['mbar']:.3e}; "
              f"min[chain - h] {w['chain']:.3e}; condition met {ncond}/150, min[W - h | cond] {w['dec']:.3e}")
