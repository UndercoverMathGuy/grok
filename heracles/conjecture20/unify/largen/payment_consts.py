# Lower bounds on payment constants from explicit slot-symmetric atom systems (beta_T = m_{|T|} = sum_i beta_i xi_i^{|T|}):
#   c1   = gain / max_T |beta_T|          (pointwise payment, Note's c_1(n,s))
#   k*   = gain / sqrt(E/2)               (L2 payment)
#   poly = gain / sup_{D_R^n} |P - A| >= gain / sum_k C(n,k) |m_k| R^k   (polydisc payment, seed S4), R = 1/2
import sys, numpy as np, mpmath as mp
from math import comb
mp.mp.dps = 80
def consts(z, m, n, s, R=0.5):
    rho, al, ph = z[:m], z[m:2*m], z[2*m:]
    beta = [mp.e**mp.mpf(float(r)) * mp.expj(mp.mpf(float(a))) for r, a in zip(rho, al)]
    xi = [mp.expj(mp.mpf(float(x))) for x in ph]
    mk = [mp.fsum([b * x**k for b, x in zip(beta, xi)]) for k in range(n+1)]
    A = mk[0]; Q = mp.fsum([abs(b)**(1/mp.mpf(s)) for b in beta]); nu = n+1
    gain = mp.re(A) - mp.mpf(nu)**(1-mp.mpf(s)) * Q**mp.mpf(s)
    E = mp.fsum([comb(n,k) * abs(mk[k])**2 for k in range(1, n+1)])
    mx = max(abs(mk[k]) for k in range(1, n+1))
    polysup_ub = mp.fsum([comb(n,k) * abs(mk[k]) * mp.mpf(R)**k for k in range(1, n+1)])
    L = mp.fsum([abs(b) for b in beta])
    return dict(gain=gain, kstar=gain/mp.sqrt(E/2), c1=gain/mx, poly=gain/polysup_ub, A=A, L=L)
for f in sys.argv[1:]:
    d = np.load(f); m = int(d[0]); z = d[1:]
    n = int(f.split('_n')[1].split('_')[0].split('.')[0]); s = (n+1)/2
    c = consts(z, m, n, s)
    print(f"{f}: n={n} gain/|A|={mp.nstr(c['gain']/abs(c['A']),4)} L/|A|={mp.nstr(c['L']/abs(c['A']),4)}  k*>={mp.nstr(c['kstar'],5)}"
          f"  c1>={mp.nstr(c['c1'],5)}  polydisc(R=1/2) const>={mp.nstr(c['poly'],5)}", flush=True)
# explicit family: m=n aligned atoms, xi_it = e^{2 pi i i/n}: P = A(1+z_1...z_n)
print("explicit family P=A(1+z_1..z_n) (n atoms, IF needs n+1):")
for n in [4, 8, 16, 32, 64]:
    s = (n+1)/2; g = 1 - (n/(n+1))**(s-1)
    print(f"  n={n}: gain/A={g:.4f}  max_T|beta_T|/A=1 -> c1>={g:.4f};  sup_(D_1/2^n)|P-A|/A=2^-n -> polydisc const>={g*2**n:.3e}")
