# Cone margin of a slot-symmetric class-C single-feature network built from atoms (beta_i, xi_i):
#   field at count vector N (over the p-th roots z = omega^a, omega = e^{4 pi i/p}):  P(N) = sum_i beta_i prod_a (1 + xi_i omega^a)^{N_a}
#   cone margin c = min_N [Re P - cot(pi/p) |Im P|];  (Q1) in class C says c <= nu^{1-s} Q^s.
import sys, math, numpy as np
sys.path.insert(0,'.')
from mfnet import counts
def cone_margin(z, m, n, p):
    rho, al, ph = z[:m], z[m:2*m], z[2*m:]
    beta = np.exp(rho + 1j*al); xi = np.exp(1j*ph)
    N, y = counts(n, p)
    om = np.exp(4j*np.pi*np.arange(p)/p)
    F = 1 + xi[:, None]*om[None, :]                       # m x p
    logF = np.log(F)                                       # principal branch is fine: exp(N log F) = prod F^N
    P = np.exp(N @ logF.T) @ beta                          # K
    cm = P.real - abs(P.imag)/math.tan(math.pi/p)
    return cm.min(), np.abs(P).max(), P
for f in sys.argv[1:]:
    d = np.load(f); m = int(d[0]); z = d[1:]
    n = int(f.split('_n')[1].split('_')[0].split('.')[0]); s = (n+1)/2
    beta = np.exp(z[:m] + 1j*z[m:2*m]); A = beta.sum()
    IF = (n+1)**(1-s) * (np.abs(beta)**(1/s)).sum()**s
    for p in [5, 7, 11]:
        if math.comb(n+p-1, p-1) > 3e6: continue
        c, Pmax, P = cone_margin(z, m, n, p)
        print(f"{f} p={p}: A={A.real:+.4f}{A.imag:+.4f}i  IF price={IF:.4f}  cone margin c={c:+.3e}  c/IF={c/IF:+.3e}  max|P|/|A|={Pmax/abs(A):.3e}", flush=True)
