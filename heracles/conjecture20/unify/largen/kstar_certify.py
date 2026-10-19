# Rigorous lower bounds for k*(n,s) (Companion note Thm 14's constant) from explicit slot-symmetric atom systems.
# The atoms are stored as float64 numbers (exact dyadic rationals): beta_i = exp(rho_i) e^{i alpha_i}, xi_i = e^{i phi_i}.
# We evaluate, in mpmath interval arithmetic, A = sum beta_i, Q = sum |beta_i|^{1/s},
#   E = sum_{k=1}^n C(n,k) |sum_i beta_i xi_i^k|^2   ( = sum over nonempty T of |beta_T|^2 for slot-symmetric atoms )
# and the ratio (Re A - nu^{1-s} Q^s)/sqrt(E/2).  Its lower endpoint is a rigorous lower bound on k*(n,s).
import sys, numpy as np
from mpmath import iv, mp

def certify(n, z, m, s, prec=200):
    iv.prec = prec
    rho, al, ph = z[:m], z[m:2*m], z[2*m:]
    nu = n + 1
    mod = [iv.exp(iv.mpf(float(r))) for r in rho]
    ca = [iv.cos(iv.mpf(float(a))) for a in al]; sa = [iv.sin(iv.mpf(float(a))) for a in al]
    bre = [mo * c for mo, c in zip(mod, ca)]; bim = [mo * s_ for mo, s_ in zip(mod, sa)]
    ReA = sum(bre); ImA = sum(bim)
    sI = iv.mpf(s)
    Q = sum(iv.exp(iv.log(mo) / sI) for mo in mod)
    IF = iv.exp((1 - sI) * iv.log(iv.mpf(nu)) + sI * iv.log(Q))
    E = iv.mpf(0)
    binom = iv.mpf(1)
    for k in range(1, n + 1):
        binom = binom * (n - k + 1) / k
        re = iv.mpf(0); im = iv.mpf(0)
        for i in range(m):
            ang = iv.mpf(float(ph[i])) * k
            c = iv.cos(ang); sn = iv.sin(ang)
            re += bre[i] * c - bim[i] * sn
            im += bre[i] * sn + bim[i] * c
        E += binom * (re * re + im * im)
    ratio = (ReA - IF) / iv.sqrt(E / 2)
    return ratio, ReA, ImA, IF, E

if __name__ == "__main__":
    for f in sys.argv[1:]:
        d = np.load(f); m = int(d[0]); z = d[1:]
        n = int(f.split('_n')[1].split('_')[0].split('.')[0])
        s = (n + 1) / 2.0
        ratio, ReA, ImA, IF, E = certify(n, z, m, s)
        lo = mp.mpf(ratio.a); hi = mp.mpf(ratio.b)
        print(f"{f}: n={n} s={s} m={m}  certified k*(n,s) >= {mp.nstr(lo, 8)}   (enclosure width {mp.nstr(hi-lo, 3)})", flush=True)
