# Single-feature payments at n = 2 (p-free atom problem).
#   atoms beta_i in C, xi_i, eta_i in T;  A = sum beta, B = sum beta xi, C = sum beta eta, D = sum beta xi eta,
#   Q = sum |beta_i|^r, theta = 3^{1-s}, R = sqrt(|B|^2+|C|^2+|D|^2), L = sum |beta_i|.
#  (P1) numerical sup of the L2 payment ratio  c(s) = (|A| - theta Q^s)/R   (Theorem A: c <= 1/(2 sqrt 3) = 0.2887;
#       single atom: (1 - 3^{1-s})/sqrt 3);
#  (P2) S_3 symmetry: the map (x,Y,Z) -> (Y,x,Z) on beta = r x Y Z, xi = conj(Y)^2, eta = conj(Z)^2 keeps |A|, Q, L, |C| and swaps |B|,|D|;
#  (P3) symmetrised quadratic payment  |A| - theta Q^s <= (2(s-1)/(3L)) (R^2 + eta_s min(|B|^2,|C|^2,|D|^2)),  eta_s = 4(s-1)/(7-4s):
#       random + adversarial check of the ratio (must be <= 1).
import numpy as np, sys
from scipy.optimize import minimize
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
def sums(z, N):
    rho = np.exp(z[:N]); ph = z[N:2*N]; a1 = z[2*N:3*N]; a2 = z[3*N:4*N]
    b = rho*np.exp(1j*ph); xi = np.exp(1j*a1); et = np.exp(1j*a2)
    return b, xi, et
def quantities(b, xi, et, s):
    r = 1/s; A = b.sum(); B = (b*xi).sum(); C = (b*et).sum(); D = (b*xi*et).sum()
    Q = (np.abs(b)**r).sum(); L = np.abs(b).sum()
    return A, B, C, D, Q, L
def l2ratio(z, N, s):
    b, xi, et = sums(z, N); A, B, C, D, Q, L = quantities(b, xi, et, s)
    R = np.sqrt(abs(B)**2 + abs(C)**2 + abs(D)**2)
    return (abs(A) - 3**(1 - s)*Q**s)/max(R, 1e-300)
def quad_ratio(z, N, s):
    b, xi, et = sums(z, N); A, B, C, D, Q, L = quantities(b, xi, et, s)
    eta = 4*(s - 1)/(7 - 4*s)
    m2 = min(abs(B)**2, abs(C)**2, abs(D)**2); R2 = abs(B)**2 + abs(C)**2 + abs(D)**2
    rhs = 2*(s - 1)/(3*L)*(R2 + eta*m2)
    return (abs(A) - 3**(1 - s)*Q**s)/max(rhs, 1e-300)
def adv(fun, s, Nmax, reps):
    best = -np.inf; arg = None
    for rep in range(reps):
        N = int(rng.integers(1, Nmax + 1))
        z = np.concatenate([rng.normal(size=N), rng.uniform(0, 2*np.pi, 3*N)])
        if rep % 3 == 0:  # near IF block
            z[N:2*N] = 0.1*rng.normal(size=N); k = np.arange(N)
            z[2*N:3*N] = 2*np.pi*k/max(N, 1) + 0.2*rng.normal(size=N); z[3*N:4*N] = 2*np.pi*k/max(N, 1) + 0.2*rng.normal(size=N)
        res = minimize(lambda w: -fun(w, N, s), z, method='Nelder-Mead', options={'maxiter': 4000, 'xatol': 1e-9, 'fatol': 1e-12})
        val = -res.fun
        if val > best: best, arg = val, (N, res.x)
    return best, arg
if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'sym':
        worst = 0
        for t in range(2000):
            N = int(rng.integers(1, 7)); s = rng.uniform(1, 1.5)
            r = rng.exponential(size=N); x = np.exp(2j*np.pi*rng.random(N)); Y = np.exp(2j*np.pi*rng.random(N)); Z = np.exp(2j*np.pi*rng.random(N))
            def q(x, Y, Z):
                b = r*x*Y*Z; xi = np.conj(Y)**2; et = np.conj(Z)**2
                A, B, C, D, Q, L = quantities(b, xi, et, s); return np.array([abs(A), abs(B), abs(C), abs(D), Q, L])
            q0 = q(x, Y, Z); q1 = q(Y, x, Z); q2 = q(Z, Y, x)
            worst = max(worst, np.abs(q1 - q0[[0, 3, 2, 1, 4, 5]]).max(), np.abs(q2 - q0[[0, 1, 3, 2, 4, 5]]).max())
        print(f"(P2) S_3 symmetry: max deviation over 2000 random systems = {worst:.2e}")
    elif mode == 'l2':
        for s in [1.1, 1.2, 1.3, 1.4, 1.5]:
            best, arg = adv(l2ratio, s, 6, int(sys.argv[3]) if len(sys.argv) > 3 else 60)
            print(f"(P1) s={s}: sup L2 payment ratio found = {best:.5f}; single atom (1-3^(1-s))/sqrt3 = {(1-3**(1-s))/np.sqrt(3):.5f}; "
                  f"Theorem A 1/(2 sqrt3) = {1/(2*np.sqrt(3)):.5f}; best N = {arg[0]}", flush=True)
    elif mode == 'quad':
        for s in [1.05, 1.1, 1.2, 1.3, 1.4, 1.5]:
            best, arg = adv(quad_ratio, s, 6, int(sys.argv[3]) if len(sys.argv) > 3 else 60)
            print(f"(P3) s={s}: sup of payment / symmetrised quadratic bound = {best:.5f} (<= 1 required); N = {arg[0]}", flush=True)
