# Referee checks for R8: S3 symmetry map, P12 transported to single-T={1} systems, symmetrised quadratic payment.
import numpy as np
from scipy.optimize import minimize
rng = np.random.default_rng(31)
def ABCD(beta, xi, eta): return beta.sum(), (beta*xi).sum(), (beta*eta).sum(), (beta*xi*eta).sum()
# (a) explicit swap map x <-> Y
dev = 0
for t in range(2000):
    N = int(rng.integers(1, 7)); beta = rng.normal(size=N) + 1j*rng.normal(size=N)
    xi = np.exp(1j*rng.uniform(0, 2*np.pi, N)); eta = np.exp(1j*rng.uniform(0, 2*np.pi, N))
    Y = np.sqrt(np.conj(xi)); Z = np.sqrt(np.conj(eta)); x = beta/np.abs(beta)/(Y*Z)
    # swap x<->Y: new atoms beta' = r x' Y' Z' with x'=Y, Y'=x: beta' = beta, xi' = conj(x)^2, eta' = eta
    A, B, C, D = ABCD(beta, xi, eta); A2, B2, C2, D2 = ABCD(beta, np.conj(x)**2, eta)
    # swap x<->Z: xi'' = xi, eta'' = conj(x)^2
    A3, B3, C3, D3 = ABCD(beta, xi, np.conj(x)**2)
    dev = max(dev, abs(abs(A2) - abs(A)), abs(abs(B2) - abs(D)), abs(abs(C2) - abs(C)), abs(abs(D2) - abs(B)),
              abs(abs(A3) - abs(A)), abs(abs(C3) - abs(D)), abs(abs(B3) - abs(B)), abs(abs(D3) - abs(C)))
print(f"(a) S3 swap maps: max deviation {dev:.2e}")
# (b) single-T={1} systems (C = D = 0): |A| - c12 |B| <= th Q^s ; adversarial with C=D=0 enforced by a 2-atom correction
def single_T_viol(z, N, s):
    th = 3**(1 - s); c12 = 1 - (2/3)**(s - 1)
    beta = z[:N] + 1j*z[N:2*N]; xi = np.exp(1j*z[2*N:3*N]); eta = np.exp(1j*z[3*N:4*N])
    # two extra atoms with eta = e^{i a}, -conj... solve beta_e (2 complex) so that C = D = 0
    e1, e2 = np.exp(1j*z[4*N]), np.exp(1j*z[4*N + 1]); x1, x2 = np.exp(1j*z[4*N + 2]), np.exp(1j*z[4*N + 3])
    M = np.array([[e1, e2], [x1*e1, x2*e2]]); rhs = -np.array([(beta*eta).sum(), (beta*xi*eta).sum()])
    try: be = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError: return -1
    bb = np.r_[beta, be]; xx = np.r_[xi, x1, x2]; ee = np.r_[eta, e1, e2]
    A, B, C, D = ABCD(bb, xx, ee); L = np.abs(bb).sum(); Q = (np.abs(bb)**(1/s)).sum()
    assert abs(C) + abs(D) < 1e-8*L
    return (abs(A) - c12*abs(B) - th*Q**s)/L
for s in [1.5, 1.25]:
    best = -np.inf
    for t in range(150):
        N = int(rng.integers(1, 5)); z = np.r_[rng.normal(size=2*N), rng.uniform(0, 2*np.pi, 2*N + 4)]
        r = minimize(lambda w: -single_T_viol(w, N, s), z, method='Nelder-Mead', options=dict(maxiter=5000))
        best = max(best, -r.fun)
    print(f"(b) s={s}: max (|A| - c12|B| - th Q^s)/L over C=D=0 systems = {best:.3e} (<= 0 required)")
# (c) symmetrised quadratic payment |A| - th Q^s <= Ksym R^2 / L
def quad_ratio(z, N, s):
    th = 3**(1 - s); eta_ = 4*(s - 1)/(7 - 4*s); K = (2*(s - 1)/3)*(1 + eta_/3)
    beta = np.exp(z[:N])*np.exp(1j*z[N:2*N]); xi = np.exp(1j*z[2*N:3*N]); eta = np.exp(1j*z[3*N:4*N])
    A, B, C, D = ABCD(beta, xi, eta); L = np.abs(beta).sum(); Q = (np.abs(beta)**(1/s)).sum()
    R2 = abs(B)**2 + abs(C)**2 + abs(D)**2
    return (abs(A) - th*Q**s)/(K*R2/L + 1e-300)
for s in [1.5, 1.3, 1.1]:
    best = -np.inf
    for t in range(200):
        N = int(rng.integers(1, 6)); z = np.r_[0.5*rng.normal(size=N), rng.uniform(0, 2*np.pi, 3*N)]
        if t % 2: z[2*N:] = np.r_[np.angle(np.exp(2j*np.pi*np.arange(N)/3)), np.angle(np.exp(2j*np.pi*np.arange(N)/3))] + 0.3*rng.normal(size=2*N); z[N:2*N] *= 0.1
        r = minimize(lambda w: -quad_ratio(w, N, s), z, method='Nelder-Mead', options=dict(maxiter=4000))
        best = max(best, -r.fun)
    print(f"(c) s={s}: max payment / (Ksym R^2/L) = {best:.4f} (<= 1 required)")
