# Network-level checks for class C at n = 2 (frequency-pure, no input DC).
#   (N1) conventions: atoms from the neuron parameters agree with lib.neuron_atoms / coeffs (table FFT);
#   (N2) Z_delta(x) = g(x,delta) - E g(.,delta) is the ridge sum D_1(2x1) + D_2(2x2) + D_12(2x1+2x2), and
#        A_delta = -min Z_delta,  A'_delta = -min Z_{-delta} satisfy
#        m_delta m_-delta >= A A' >= E_delta - 2/(p+1) sum_{T<T'} sqrt(V_T V_T') >= (p-1)/(p+1) E_delta,
#        E_delta = sum_{l,T} |beta_{l,T}|^2 kappa_l(delta) = sum_T Var D_T;
#   (N3) a class-C network realising the ramp profile has A A' / E = (p-1)/(p+1) exactly (sharpness at network level);
#   (N4) decoupling chain on networks with cheap features:
#        h <= sum_l a_l k_l + c_A sum_{L'} R_l k_l - mbar   (Theorem A, c_A = 1/(2 sqrt 3)),  k_l = kappa_l(delta*),
#        and, when c_A^2 sum_{L'} k_l < (p-1)/(p+1):  h <= W(theta) := min_delta sum_l a_l kappa_l(delta).
import numpy as np, itertools, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gibbs'))
from lib import table, gaps, coeffs, neuron_atoms, kappa
rng = np.random.default_rng(7)
TS = [(1, 0), (0, 1), (1, 1)]

def neuron(p, l, al, phi, psi):
    """class-C neuron of frequency l: u_t = al[t] cos(2 pi l x/p + phi[t]) (t=1,2), w = al[0] cos(2 pi l c/p + psi)."""
    a = np.arange(p)
    U = np.array([al[1]*np.cos(2*np.pi*l*a/p + phi[0]), al[2]*np.cos(2*np.pi*l*a/p + phi[1])])
    W = al[0]*np.cos(2*np.pi*l*a/p + psi)
    return U, W

def net_from_atoms(p, atoms):
    """atoms: list of (l, beta (complex), xi1, xi2 (unit complex)) -> (U, W) with balanced norms."""
    Us, Ws, freqs = [], [], []
    for (l, b, x1, x2) in atoms:
        phi = [-np.angle(x1)/2, -np.angle(x2)/2]
        psi = phi[0] + phi[1] - np.angle(b)
        al = (4*abs(b))**(1/3)
        U, W = neuron(p, l, [al, al, al], phi, psi)
        Us.append(U); Ws.append(W); freqs.append(l)
    return np.array(Us), np.array(Ws), np.array(freqs)

def solve_atoms(target, xis):
    """target (A, B, C, D) = sum_i beta_i (1, xi_i1, xi_i2, xi_i1 xi_i2): solve for 4 atoms on the xi grid."""
    Mx = np.array([[1, x1, x2, x1*x2] for (x1, x2) in xis]).T
    return np.linalg.solve(Mx, np.array(target))

def analyse(U, W, freqs, p, b=2.0):
    n = 2; nu = 3; s = nu/b; r = 1/s
    f = table(U, W); g = gaps(f); h = g[..., 1:].min()
    co = coeffs(f); Fn = (p - 1)//2
    beta_i, xi = neuron_atoms(U, W)
    Q = np.sum(np.abs(beta_i)**r, axis=0); a = nu**(1 - s)*Q**s
    beta = np.array([co[(l, (0, 0))] for l in range(1, Fn + 1)])
    bT = np.array([[co[(l, T)] for T in TS] for l in range(1, Fn + 1)])       # (Fn, 3)
    R = np.sqrt((np.abs(bT)**2).sum(1))
    K = np.array([[kappa(l, d, p) for l in range(1, Fn + 1)] for d in range(p)])  # K[d, l-1]
    Wd = (K[1:] @ a); dstar = int(np.argmin(Wd)) + 1
    mean_g = g.reshape(-1, p).mean(0)
    m = mean_g - h
    out = dict(h=h, beta=beta, bT=bT, R=R, a=a, Q=Q, K=K, dstar=dstar, m=m, W=Wd.min(), g=g)
    return out

def ridge_check(out, p, d):
    g = out['g']; Z = g[..., d] - g[..., d].mean(); Zm = g[..., (-d) % p] - g[..., (-d) % p].mean()
    x1, x2 = np.meshgrid(np.arange(p), np.arange(p), indexing='ij')
    inv2 = pow(2, -1, p)
    # D_T(w) = E[Z | 2 s_T(x) = w]
    D = []
    for T in TS:
        sT = (T[0]*x1 + T[1]*x2) % p
        D.append(np.array([Z[(2*sT) % p == w].mean() for w in range(p)]))
    rec = D[0][(2*x1) % p] + D[1][(2*x2) % p] + D[2][(2*(x1 + x2)) % p]
    ridge_err = np.abs(rec - Z).max()
    # Z_{-d}(x) = - sum_T D_T(2 s_T(x) + d)
    recm = -(D[0][(2*x1 + d) % p] + D[1][(2*x2 + d) % p] + D[2][(2*(x1 + x2) + d) % p])
    refl_err = np.abs(recm - Zm).max()
    VT = np.array([(Dt**2).mean() for Dt in D]); E = VT.sum()
    Efor = (np.abs(out['bT'])**2 * out['K'][d][:, None]).sum()
    A = -Z.min(); Ap = -Zm.min()
    lower = E - 2/(p + 1)*sum(np.sqrt(VT[i]*VT[j]) for i, j in [(0, 1), (0, 2), (1, 2)])
    return dict(ridge_err=ridge_err, refl_err=refl_err, E=E, E_err=abs(E - Efor), A=A, Ap=Ap, lower=lower,
                mm=out['m'][d]*out['m'][(-d) % p])

def random_classC(p, kmax=3, scale_spread=1.0):
    atoms = []
    for l in range(1, (p - 1)//2 + 1):
        for _ in range(int(rng.integers(1, kmax + 1))):
            atoms.append((l, rng.exponential()*np.exp(1j*rng.normal()*scale_spread),
                          np.exp(2j*np.pi*rng.random()), np.exp(2j*np.pi*rng.random())))
    return net_from_atoms(p, atoms)

if __name__ == '__main__':
    cA = 1/(2*np.sqrt(3))
    # (N1)-(N2) random class-C networks
    for p in [5, 7, 11, 13]:
        worst = dict(ridge=0, refl=0, Eerr=0, AA=np.inf, mm=np.inf, conv=0)
        for t in range(25):
            U, W, fr = random_classC(p)
            out = analyse(U, W, fr, p)
            # (N1) atom conventions: product formula vs lib
            for i, l in enumerate(fr):
                pass
            for d in range(1, p):
                c = ridge_check(out, p, d)
                worst['ridge'] = max(worst['ridge'], c['ridge_err']); worst['refl'] = max(worst['refl'], c['refl_err'])
                worst['Eerr'] = max(worst['Eerr'], c['E_err']/max(c['E'], 1e-300))
                worst['AA'] = min(worst['AA'], (c['A']*c['Ap'] - c['lower'])/c['E'])
                worst['mm'] = min(worst['mm'], (c['mm'] - c['A']*c['Ap'])/c['E'])
        print(f"(N2) p={p}: ridge-sum err {worst['ridge']:.1e}, reflection err {worst['refl']:.1e}, E formula rel err {worst['Eerr']:.1e}, "
              f"min (AA' - bound)/E = {worst['AA']:.3e} (>=0), min (m m' - AA')/E = {worst['mm']:.3e} (>=0)", flush=True)
    # (N3) ramp realisation
    for p in [5, 7, 11, 13, 17]:
        d = 1
        r = ((-np.arange(p)) % p).astype(float); r -= r.mean()
        Dtarget = [r, r, r[(1 - np.arange(p)) % p]]     # f1 = D_1, f2 = D_2, f3(z) = D_12(-z) = ramp(z+1): puts the ramp's worst planes (1,2) at (0,d=1)
        atoms = []
        xis = [(1, 1), (1, 1j), (1j, 1), (1j, 1j)]
        for l in range(1, (p - 1)//2 + 1):
            # a_{l,T} = 2 Dhat_T(l) / (p (1 - rho^{-l d})),  beta_{l,T} = conj(a_{l,T})
            bt = []
            for T, Dt in zip(TS, Dtarget):
                Dh = np.sum(Dt*np.exp(-2j*np.pi*l*np.arange(p)/p))
                bt.append(np.conj(2*Dh/(p*(1 - np.exp(-2j*np.pi*l*d/p)))))
            beta_clean = 3.0*max(abs(x) for x in bt)    # any clean amplitude; does not enter Z
            sol = solve_atoms([beta_clean, bt[0], bt[1], bt[2]], xis)
            atoms += [(l, sol[i], xis[i][0], xis[i][1]) for i in range(4)]
        U, W, fr = net_from_atoms(p, atoms)
        out = analyse(U, W, fr, p)
        c = ridge_check(out, p, d)
        print(f"(N3) ramp network p={p}: A = {c['A']:.6f}, A' = {c['Ap']:.6f}, E = {c['E']:.6f}, AA'/E = {c['A']*c['Ap']/c['E']:.8f}, "
              f"(p-1)/(p+1) = {(p-1)/(p+1):.8f}", flush=True)
    # (N4) decoupling chain on networks near IF with cheap features
    for p in [7, 11, 13, 17, 19, 23]:
        worst_chain = np.inf; worst_dec = np.inf; ncheap = []; hpos = 0
        for t in range(40):
            atoms = []
            for l in range(1, (p - 1)//2 + 1):
                if rng.random() < 0.5: continue
                amp = rng.exponential()
                if rng.random() < 0.5:     # cheap realisation: 1-2 atoms with spread sign data
                    k = int(rng.integers(1, 3))
                    for _ in range(k):
                        atoms.append((l, amp/k*np.exp(0.2j*rng.normal()), np.exp(2j*np.pi*rng.random()), np.exp(2j*np.pi*rng.random())))
                else:                      # IF-like block (3 atoms, xi = e^{2 pi i k/3}) plus a small perturbation
                    for k3 in range(3):
                        z = np.exp(2j*np.pi*k3/3)
                        atoms.append((l, amp/3*(1 + 0.1*rng.normal()), z*np.exp(0.1j*rng.normal()), z*np.exp(0.1j*rng.normal())))
            if not atoms: continue
            U, W, fr = net_from_atoms(p, atoms)
            out = analyse(U, W, fr, p, b=2.0)
            dst = out['dstar']; k = out['K'][dst]
            cheap = np.abs(out['beta']) > out['a'] + 1e-12
            ncheap.append(int(cheap.sum()))
            mbar = 0.5*(out['m'][dst] + out['m'][(-dst) % p])
            rhs = (out['a']*k).sum() + cA*(out['R'][cheap]*k[cheap]).sum() - mbar
            worst_chain = min(worst_chain, rhs - out['h'])
            if cA**2*k[cheap].sum() < (p - 1)/(p + 1):
                worst_dec = min(worst_dec, out['W'] - out['h'])
            hpos += out['h'] > 0
        print(f"(N4) p={p}: min[chain RHS - h] = {worst_chain:.3e} (>=0);  min[W - h] over nets meeting the condition = {worst_dec:.3e} (>=0);  "
              f"cheap counts {min(ncheap)}..{max(ncheap)}, nets with h>0: {hpos}/40", flush=True)
