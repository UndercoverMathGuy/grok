# Referee: independent re-verification of the R9(a) compression certificates (float64 with explicit, generous error
# allowance; uniform grid; different code path from the agent's adaptive mpmath scheme).
#   g(x) = sum_l b'_l (1 - cos l x) - P_q(x) >= 0 on [0, pi];  rho_q = (sum b'^{2/3})^{3/2} / q^{3/2} < (2/3)^{1/2}.
# [0, x_s]: g >= x^2/2 (S2' - S2) - x^4 S4'/24 >= 0 for x <= x_s = sqrt(12 (S2'-S2)/S4');
# [x_s, pi]: N uniform cells, g >= min(endpoints) - M2 h^2/8 - err,  M2 = S2' + S2,  err = 1e-9 (float allowance).
# P_q is computed by DIRECT summation sum_j (1 - cos j x) in chunks (not the closed form), as a further independent path.
import json, numpy as np, sys
from fractions import Fraction
D = '/home/user/grok/heracles/conjecture20/unify/arith2/'
thr = np.sqrt(2/3)
def check(q, freqs, bstr, N=2_000_000):
    b = np.array([float(Fraction(x)) for x in bstr]); l = np.array(freqs, float)
    S2p = (b*l**2).sum(); S4p = (b*l**4).sum(); S2 = q*(q+1)*(2*q+1)/6
    if S2p <= S2: return False, 'S2', None
    xs = np.sqrt(12*(S2p-S2)/S4p)*(1-1e-12)
    x = np.linspace(xs, np.pi, N+1); h = x[1]-x[0]
    g = np.zeros_like(x)
    for bl, ll in zip(b, l): g += bl*(1-np.cos(ll*x))
    for j0 in range(1, q+1, 64):
        js = np.arange(j0, min(q, j0+63)+1, dtype=float)
        g -= (1-np.cos(np.outer(x, js))).sum(1)
    M2 = S2p+S2
    lb = np.minimum(g[:-1], g[1:]) - M2*h*h/8 - 1e-9
    rho = (np.sum(b**(2/3)))**1.5/q**1.5
    return lb.min() >= 0, lb.min(), rho
if __name__ == '__main__':
    want = [int(a) for a in sys.argv[1].split(',')] if len(sys.argv) > 1 else None
    seen = {}
    for fn in ['compress_cert_uniform_s1.5.json', 'compress_cert_warm_s1.5.json', 'compress_cert_big_s1.5.json']:
        for q, e in json.load(open(D+fn)).items():
            q = int(q)
            if want and q not in want: continue
            ok, lbm, rho = check(q, e['freqs'], e['bprime'], N=400_000 if q <= 64 else 1_000_000)
            print(f"{fn[14:19]} q={q}: g>=0 verified={ok} (min cell lb {lbm:.3e}); rho={rho:.5f} ({'below' if rho < thr else 'NOT below'} {thr:.5f})", flush=True)
