# Independent re-certification of largen R5: lower bounds on k*(n, nu/2) from the stored slot-symmetric atom systems.
# k*(n,s) = sup over atoms of (Re A - nu^{1-s} Q^s)/sqrt(E/2), A = sum beta_i, Q = sum |beta_i|^{1/s},
# E = sum_{T nonempty} |sum_i beta_i xi_i^T|^2.  For slot-symmetric atoms xi_i^T = xi_i^{|T|}.
# Independent formula used here (Gram form, real interval arithmetic only):
#   E = sum_{i,j} |b_i||b_j| [ (2cos(d_ij/2))^n cos(a_i - a_j + n d_ij/2) - cos(a_i - a_j) ],  d_ij = phi_i - phi_j,
# which follows from sum_{T} xi_i^T conj(xi_j)^T = prod_t (1 + e^{i d_ij}) = (2cos(d/2))^n e^{i n d/2} (T over all subsets).
# Plus: brute-force over all 2^n - 1 subsets for n=9 (mp floats) as a cross-check of the slot-symmetric reduction.
import glob, itertools, os
import numpy as np
from mpmath import iv, mp
iv.prec = 260; mp.prec = 260
D = '/home/user/grok/heracles/conjecture20/unify/largen/'

def load(f):
    d = np.load(f); m = int(d[0]); z = d[1:]
    assert len(z) == 3 * m, (f, len(z), m)
    return m, z[:m], z[m:2 * m], z[2 * m:]

def kstar_iv(n, rho, al, ph):
    nu = n + 1; s = iv.mpf(nu) / 2; m = len(rho)
    R = [iv.mpf(float(x)) for x in rho]; Al = [iv.mpf(float(x)) for x in al]; Ph = [iv.mpf(float(x)) for x in ph]
    mod = [iv.exp(r) for r in R]
    ReA = sum(mo * iv.cos(a) for mo, a in zip(mod, Al))
    Q = sum(iv.exp(r / s) for r in R)
    IF = iv.exp((1 - s) * iv.log(iv.mpf(nu)) + s * iv.log(Q))
    E = iv.mpf(0)
    for i in range(m):
        for j in range(m):
            d = Ph[i] - Ph[j]; da = Al[i] - Al[j]
            twoc = 2 * iv.cos(d / 2)
            E += mod[i] * mod[j] * (twoc ** n * iv.cos(da + n * d / 2) - iv.cos(da))
    return (ReA - IF) / iv.sqrt(E / 2), ReA, IF, E

def kstar_bruteforce(n, rho, al, ph):
    nu = n + 1; s = mp.mpf(nu) / 2
    beta = [mp.e ** mp.mpf(float(r)) * mp.expj(mp.mpf(float(a))) for r, a in zip(rho, al)]
    xi = [mp.expj(mp.mpf(float(x))) for x in ph]
    A = mp.fsum(beta); Q = mp.fsum([abs(b) ** (1 / s) for b in beta])
    E = mp.mpf(0)
    for k in range(1, n + 1):
        mk = mp.fsum([b * x ** k for b, x in zip(beta, xi)])
        for T in itertools.combinations(range(n), k):
            E += abs(mk) ** 2
    return (mp.re(A) - mp.mpf(nu) ** (1 - s) * Q ** s) / mp.sqrt(E / 2)

if __name__ == '__main__':
    files = sorted(glob.glob(D + 'kstar_opt_n*_b2.0.npy') + glob.glob(D + 'comb_opt_n*_b2.0.npy') + glob.glob(D + 'arc*_n*.npy'),
                   key=lambda f: int(os.path.basename(f).split('_n')[1].split('_')[0].split('.')[0]))
    k5 = iv.sqrt(iv.cot(iv.pi / 5) ** 2 - 1); k7 = iv.sqrt(iv.cot(iv.pi / 7) ** 2 - 1); k11 = iv.sqrt(iv.cot(iv.pi / 11) ** 2 - 1)
    print(f"k_5 in [{mp.nstr(k5.a,8)},{mp.nstr(k5.b,8)}], k_7 in [{mp.nstr(k7.a,8)},{mp.nstr(k7.b,8)}], k_11 in [{mp.nstr(k11.a,8)},{mp.nstr(k11.b,8)}]")
    for f in files:
        n = int(os.path.basename(f).split('_n')[1].split('_')[0].split('.')[0])
        m, rho, al, ph = load(f)
        r, ReA, IF, E = kstar_iv(n, rho, al, ph)
        extra = ''
        if n == 9:
            extra = f'  brute-force over subsets: {mp.nstr(kstar_bruteforce(n, rho, al, ph), 10)}'
        sep = len(set(np.round(np.mod(ph, 2 * np.pi), 12)))
        print(f"{os.path.basename(f)}: n={n} m={m} (distinct phases {sep}) k* >= [{mp.nstr(r.a, 10)}, {mp.nstr(r.b, 10)}]"
              f"  > k_5:{r.a > k5.b}  > k_7:{r.a > k7.b}  E/ReA^2~{mp.nstr(E.a/ReA.a**2, 4)}{extra}", flush=True)
