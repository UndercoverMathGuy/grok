# Class C, n=2, worst case of  Pay^2 / Cert  (phase-free), solved as convex programs.
#  Cert constraints (each is a proved lower bound for A_d A_{-d} <= m_d m_{-d} <= mbar_d^2):
#     per-T Bhatia-Davis:            V_T <= 1
#     translation couplings, a in Z_p^2:  Q(a) = sum_{l,T} w_{l,T} k_l cos(2 pi l (d + 2 s_T(a))/p) <= 1
#  (w = cheap + noncheap interference moduli squared).  Payment over cheap features:
#     mode A:   sum_l k_l cA R_l
#     mode min: sum_l k_l min(cA R_l, c1 sqrt(x_{l,tau(l)}))  maximised over dominant-T assignments tau (each is concave)
#  Worst R = (max Pay)^2 subject to Cert <= 1.  Closure of class C at n=2 for prime p iff worst R <= 1.
import numpy as np, sys, itertools
from scipy.optimize import minimize
p = int(sys.argv[1]); mode = sys.argv[2]
F = np.arange(1, (p-1)//2 + 1); K = len(F); d = 1
k = 1 - np.cos(2*np.pi*F*d/p)
c1 = 1 - 3**(-0.5); cA = 1/(2*np.sqrt(3))
Tvec = np.array([[1,0],[0,1],[1,1]])
# linear constraint rows: each row r gives sum_{T,l} coef[r,T,l] * w[T,l] <= 1
rows = []
for T in range(3):
    c = np.zeros((3, K)); c[T] = k; rows.append(c)
for a1 in range(p):
    for a2 in range(p):
        sT = (Tvec @ np.array([a1, a2])) % p
        c = k[None, :]*np.cos(2*np.pi*np.outer(d + 2*sT, F)/p)
        rows.append(c)
Cmat = np.array([r.ravel() for r in rows])          # (nrows, 3K)
nv = 3*K
def solve(tau):
    # variables: x (3K cheap), y (3K noncheap), t (K)
    def unpack(z): return z[:nv], z[nv:2*nv], z[2*nv:]
    obj = lambda z: -(k*unpack(z)[2]).sum()
    jac = lambda z: np.concatenate([np.zeros(2*nv), -k])
    cons = [{'type': 'ineq', 'fun': lambda z: 1 - Cmat @ (unpack(z)[0] + unpack(z)[1]),
             'jac': lambda z: np.hstack([-Cmat, -Cmat, np.zeros((len(Cmat), K))])}]
    def tA(z):
        x, y, t = unpack(z); X = x.reshape(3, K).sum(0); return cA*np.sqrt(X + 1e-14) - t
    cons.append({'type': 'ineq', 'fun': tA})
    if tau is not None:
        def tC(z):
            x, y, t = unpack(z); xx = x.reshape(3, K); return c1*np.sqrt(xx[tau, np.arange(K)] + 1e-14) - t
        cons.append({'type': 'ineq', 'fun': tC})
    best = None
    for s in range(int(__import__("os").environ.get("NREST","6"))):
        rng = np.random.default_rng(s)
        z0 = np.concatenate([rng.random(nv)*0.1, rng.random(nv)*0.01*(s % 2), np.zeros(K)])
        res = minimize(obj, z0, jac=jac, constraints=cons, bounds=[(0, None)]*(2*nv) + [(0, None)]*K,
                       method='SLSQP', options={'maxiter': 2000, 'ftol': 1e-12})
        if res.success or True:
            val = -res.fun
            x, y, t = unpack(res.x)
            feas = (1 - Cmat @ (x + y)).min()
            if feas > -1e-7 and (best is None or val > best[0]): best = (val, res.x)
    return best
if mode == 'A':
    b = solve(None); R = b[0]**2
    x = b[1][:nv].reshape(3, K); y = b[1][nv:2*nv].reshape(3, K)
    print(f"p={p} mode=A: worst R = {R:.4f} -> {'CLOSED' if R <= 1 else 'not closed'}")
    np.set_printoptions(precision=3, suppress=True); print("kappa", k); print("x", x/x.max()); print("y", y/max(x.max(), 1e-12))
else:
    bestR = 0; arg = None
    # by the slot symmetry {1}<->{2}, fix tau(feature with largest kappa) in {0,2}
    for tau in itertools.product(range(3), repeat=K):
        b = solve(np.array(tau))
        if b is not None and b[0]**2 > bestR: bestR = b[0]**2; arg = (tau, b)
    print(f"p={p} mode=min: worst R = {bestR:.4f} -> {'CLOSED' if bestR <= 1 else 'not closed'}  worst tau={arg[0]}")
    x = arg[1][1][:nv].reshape(3, K); np.set_printoptions(precision=3, suppress=True); print("kappa", k); print("x", x/x.max())
