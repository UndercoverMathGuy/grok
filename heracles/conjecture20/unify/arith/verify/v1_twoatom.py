# Referee: R1 two-atom cap, random + adversarial (Nelder-Mead) attempts to break
#  (i) max|beta_i| <= (L+m)/2, (ii) Q >= ((L+m)/2)^r + ((L-m)/2)^r, m = min(|B|,|C|,|D|),
#  (iii) on B=0 systems: L <= 2^{1-s} Q^s;  plus the equality case and 1 + c tau' = tau'.
import numpy as np
from scipy.optimize import minimize
rng = np.random.default_rng(11)
def stats(x, k, s):
    r = 1/s; mag = np.abs(x[:k]); ph = x[k:2*k]; xi = np.exp(1j*x[2*k:3*k]); eta = np.exp(1j*x[3*k:4*k])
    beta = mag*np.exp(1j*ph); L = mag.sum(); Q = (mag**r).sum()
    B = (beta*xi).sum(); C = (beta*eta).sum(); D = (beta*xi*eta).sum(); m = min(abs(B), abs(C), abs(D))
    return mag, L, Q, m, r
w1 = w2 = w3 = -np.inf
for trial in range(3000):
    k = rng.integers(1, 9); s = rng.uniform(1, 1.5); x = rng.normal(size=4*k)
    mag, L, Q, m, r = stats(x, k, s)
    w1 = max(w1, mag.max()-(L+m)/2); w2 = max(w2, ((L+m)/2)**r+(max(L-m, 0)/2)**r-Q)
for trial in range(60):
    k = rng.integers(2, 7); s = rng.uniform(1, 1.5)
    f = lambda x: -(lambda mag, L, Q, m, r: (((L+m)/2)**r+(max(L-m, 0)/2)**r-Q)/max(Q, 1e-12))(*stats(x, k, s))
    res = minimize(f, rng.normal(size=4*k), method='Nelder-Mead', options=dict(maxiter=4000))
    w2 = max(w2, -res.fun)
    # B=0 systems: enforce by projecting beta*xi to zero sum (choose xi = conj(phase of beta)*unit vectors of zero-sum z)
    def g(x):
        z = x[:k]+1j*x[k:2*k]; z = z-z.mean(); mag = np.abs(z)+1e-12; L = mag.sum(); Q = (mag**(1/s)).sum()
        return -(L/(2**(1-s)*Q**s))
    res = minimize(g, rng.normal(size=2*k), method='Nelder-Mead', options=dict(maxiter=4000))
    w3 = max(w3, -res.fun-1)
print(f"(i) max[max|beta| - (L+m)/2] = {w1:.2e};  (ii) max relative violation = {w2:.2e};  (iii) max L/(2^(1-s)Q^s) - 1 on B=0 = {w3:.2e}")
s = 1.5; r = 2/3; beta = np.array([.5, .5]); xi = np.array([1j, -1j]); eta = xi
A = beta.sum(); B = (beta*xi).sum(); C = (beta*eta).sum(); D = (beta*xi*eta).sum(); Q = (np.abs(beta)**r).sum()
c = 1-(2/3)**(s-1); a = 3**(1-s)*Q**s
print(f"equality case: |A|={abs(A)}, |B|={abs(B)}, |C|={abs(C)}, |D|={abs(D)}, 2^(1-s)Q^s={2**(1-s)*Q**s:.15f}, |A|-c|D|-a={abs(A)-c*abs(D)-a:.1e}")
