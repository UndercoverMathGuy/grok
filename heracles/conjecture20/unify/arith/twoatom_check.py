# Check the two-atom cap.
# Claim (robust form): for atoms beta_i in C, xi_i in T, with L = sum|beta_i|, Q = sum|beta_i|^r, B = sum beta_i xi_i:
#   max_i |beta_i| <= (L+|B|)/2   and hence   Q >= ((L+|B|)/2)^r + ((L-|B|)/2)^r.
# Consequence at B=0 (relations case): L <= 2^{1-s} Q^s, so |A|,|D| <= L <= (3/2)^{s-1} * 3^{1-s} Q^s.
# By slot symmetry the same holds with |B| replaced by min(|B|,|C|,|D|).
import numpy as np
rng = np.random.default_rng(1)
worst_atom = -1e9; worst_Q = -1e9; worst_cap = -1e9
for trial in range(200000):
    n = rng.integers(1, 9)
    s = rng.uniform(1.0, 1.5); r = 1/s
    rho = rng.exponential(1, n) ** rng.uniform(0.2, 3)
    ph = rng.uniform(0, 2*np.pi, n); xi = np.exp(1j*rng.uniform(0, 2*np.pi, n)); eta = np.exp(1j*rng.uniform(0, 2*np.pi, n))
    beta = rho*np.exp(1j*ph)
    if trial % 2 == 0 and n >= 2:
        # project to B = 0 exactly: adjust the last atom's slot phase/magnitude if possible
        rest = np.sum(beta[:-1]*xi[:-1])
        # need beta_n xi_n = -rest: set |beta_n| = |rest|, xi_n = -rest/beta_n
        beta[-1] = abs(rest)*np.exp(1j*ph[-1]); xi[-1] = -rest/beta[-1] if abs(rest) > 0 else 1
        rho = np.abs(beta)
    L = rho.sum(); Q = (rho**r).sum(); A = beta.sum(); B = (beta*xi).sum(); C = (beta*eta).sum(); D = (beta*xi*eta).sum()
    m = min(abs(B), abs(C), abs(D))
    # max atom bound
    worst_atom = max(worst_atom, rho.max() - (L + abs(B))/2)
    # robust Q bound with min over slot-sets
    lb = ((L+m)/2)**r + ((L-m)/2)**r
    worst_Q = max(worst_Q, lb - Q)
    if abs(B) < 1e-9 * L:
        worst_cap = max(worst_cap, max(abs(A), abs(D)) / (2**(1-s) * Q**s) - 1)
print("max_i |beta_i| - (L+|B|)/2   (must be <= 0):", worst_atom)
print("((L+m)/2)^r+((L-m)/2)^r - Q, m=min_T|beta_T|  (must be <= 0):", worst_Q)
print("max(|A|,|D|)/(2^{1-s}Q^s) - 1 on B=0 systems (must be <= 0):", worst_cap)
# equality: two aligned atoms beta_1=beta_2, xi=(i,-i), eta=(i,-i)
s = 1.5; r = 1/s
beta = np.array([0.5, 0.5]); xi = np.array([1j, -1j]); eta = np.array([1j, -1j])
L = 1; Q = (np.abs(beta)**r).sum()
print("two aligned atoms: |A|=", abs(beta.sum()), " |D|=", abs((beta*xi*eta).sum()), " 2^{1-s}Q^s=", 2**(1-s)*Q**s,
      " bonus/(c12*|D|)=", (abs(beta.sum()) - 3**(1-s)*Q**s)/((1-(2/3)**(s-1))*abs((beta*xi*eta).sum())))
