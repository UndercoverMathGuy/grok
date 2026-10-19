"""Precise analysis of the adversarial atom system where the coarse float twisted bound exceeded P_1."""
import pickle, numpy as np, sys
from scipy.optimize import minimize
sys.path.insert(0, '..')
s = 1.5; r = 1/s
res = pickle.load(open('v3_results.pkl', 'rb'))
def unpack(v, k):
    rr = np.abs(v[:k]); pa = v[k:2*k]; xa = v[2*k:3*k]; ea = v[3*k:4*k]
    return rr*np.exp(1j*pa), np.exp(1j*xa), np.exp(1j*ea)
def S_roots(P, q):
    rts = np.roots([1, 0, -P, -q]).real; return np.sum(np.maximum(1+rts, 0)**r)
def Smin(P, q):
    P = min(max(P, 0), 3); dh = 2*(P/3)**1.5
    if q > dh + 1e-15: return np.inf
    return S_roots(P, max(q, P-1, -dh))
def chain_val(L, psi, mX, mY, mZ, phi):
    rho = np.sqrt(max(2*L*(L-1.0), 0))
    P = (mY**2+mZ**2+mX**2+rho**2+2*mX*rho*np.cos(psi-phi))/L**2
    q = 2*mY*mZ*(mX*np.cos(phi)+rho*np.cos(psi))/L**3
    return L*(Smin(P, q)/3)**s
print("first entries:", [round(z[0], 6) for z in res[:4]])
for idx in range(3):
    val, k, x = res[idx]
    beta, xi, eta = unpack(x, k)
    A = beta.sum(); B = (beta*xi).sum(); C = (beta*eta).sum(); D = (beta*xi*eta).sum()
    lam = abs(A); A, B, C, D = A/lam, B/lam, C/lam, D/lam; beta = beta/lam
    phi = np.angle(A*D*np.conj(B*C)); m = (abs(B), abs(C), abs(D))
    P1 = 3**(1-s)*np.sum(abs(beta)**r)**s; Ltrue = abs(beta).sum()
    print(f"\n#{idx}: k={k} |beta_i|={np.round(abs(beta),5)} arg(beta_i/A)={np.round(np.angle(beta/A),4)}")
    print(f"   m={np.round(m,6)} phi_pay={phi:.6f}  P1={P1:.6f}  true L={Ltrue:.6f}")
    # true H for each chain and its exact Rotfel'd value
    p_ = beta/abs(beta); vr = abs(beta); al = A/abs(A)
    for name, U in [('chain1', np.stack([np.ones(k), p_*xi, p_**2*xi*eta], 1)), ('chain2', np.stack([np.ones(k), p_*xi, p_*xi*eta], 1)), ('chain3', np.stack([np.ones(k), p_*eta, p_*xi*eta], 1))]:
        H = sum(vr[i]*np.outer(U[i], U[i].conj()) for i in range(k))
        ev = np.linalg.eigvalsh(H/Ltrue)
        rot = Ltrue*(np.sum(np.maximum(ev, 0)**r)/3)**s
        print(f"   {name}: Rotfel'd value L M_r(nu) = {rot:.6f} (<= P1 required)")
    best_chains = []
    for (mX, mY, mZ, nm) in [(m[2], m[0], m[1], 'X=D'), (m[1], m[0], m[2], 'X=C'), (m[0], m[1], m[2], 'X=B')]:
        Lmin = max(1, *m)
        # fine grid
        Ls = Lmin*(1+np.concatenate([[0], np.geomspace(1e-8, 1.5, 600)]))
        ps = np.linspace(-np.pi, np.pi, 1441)
        best = (np.inf, None)
        for L in Ls:
            vals = [chain_val(L, psi, mX, mY, mZ, phi) for psi in ps[::8]]
            j = int(np.argmin(vals))
            if vals[j] < best[0]: best = (vals[j], (L, ps[::8][j]))
        # local refinement
        f = lambda z: chain_val(max(z[0], Lmin), z[1], mX, mY, mZ, phi)
        rr_ = minimize(f, np.array(best[1]), method='Nelder-Mead', options=dict(xatol=1e-12, fatol=1e-14, maxiter=4000))
        mn = min(best[0], rr_.fun, 3**(1-s)*2.5*Lmin)
        best_chains.append(mn)
        print(f"   {nm}: min over (L,psi) ~ {mn:.6f} at L={rr_.x[0]:.6f}, psi={rr_.x[1]:.5f}")
    print(f"   twisted bound (max over chains) ~ {max(best_chains):.6f}   vs P1 = {P1:.6f}")
    from pay_adapt import payment
    Fr = payment(np.array(m), np.array(m), phi, phi)
    print(f"   rigorous payment (point box) = {Fr:.6f}")
