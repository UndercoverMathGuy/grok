# p=3 tests (exact rational networks of exact_certificate.py, re-entered independently by parsing that file's data).
#  (a) gamma > gamma_IF at b=2 (exact Fractions);  (b) neuron cost shares, entropy H vs log(width) vs log(nu)
#      (R3/R4 require any IF-beater's optimal version to be 'concentrated': H < log nu);
#  (c) R1 integrated form for this fixed network: log(gamma/gamma_IF)(s) is affine in s iff H(s) is constant;
#  (d) R13's optimality inequality with gamma* replaced by gamma(theta_cex): single-neuron removals.
import ast, math, re
from fractions import Fraction as Fr
import mpmath as mp

src = open('/home/user/grok/heracles/conjecture20/exact_certificate.py').read()
blk = src[src.index('NETWORKS = {'):src.index('\n\n\ndef check')]
NETWORKS = ast.literal_eval(re.sub(r'#.*', '', blk[len('NETWORKS = '):]))
mp.mp.dps = 40
M = lambda x: mp.mpf(x.numerator) / x.denominator


def margin_counts(U, W, n, idx=None):
    p = 3; m = len(U); idx = list(range(m)) if idx is None else idx; h = None; arg = None
    for N0 in range(n + 1):
        for N1 in range(n + 1 - N0):
            N2 = n - N0 - N1; y = (N1 + 2 * N2) % p
            z = {i: U[i][0] ** N0 * U[i][1] ** N1 * U[i][2] ** N2 for i in idx}
            f = [sum(W[i][c] * z[i] for i in idx) for c in range(p)]
            for c in range(p):
                if c != y and (h is None or f[y] - f[c] < h):
                    h, arg = f[y] - f[c], (N0, N1, N2, c)
    return h, arg


def gap_at(U, W, idx, arg):
    N0, N1, N2, c = arg; y = (N1 + 2 * N2) % 3
    z = {i: U[i][0] ** N0 * U[i][1] ** N1 * U[i][2] ** N2 for i in idx}
    return sum((W[i][y] - W[i][c]) * z[i] for i in idx)


for n, net in NETWORKS.items():
    nu = n + 1; U = [[Fr(a) for a in r] for r in net['u']]; W = [[Fr(a) for a in r] for r in net['w']]; m = len(U)
    h, arg = margin_counts(U, W, n)
    S = [n * sum(a * a for a in U[i]) + sum(a * a for a in W[i]) for i in range(m)]     # ||omega_i||^2
    P2 = sum(S); gam = h / P2 ** (nu // 2); gIF = Fr(3 * nu) / Fr(6 * nu * nu) ** (nu // 2)
    print(f"n={n} width={m} nu={nu}: exact gamma/gamma_IF at b=2 = {float(gam / gIF):.8f} (>1: {gam > gIF})")
    # balance each neuron (does not change table, lowers cost) and look at cost shares
    Sb = []
    for i in range(m):
        nu_ = mp.mpf(nu); tu = mp.sqrt(sum(M(a) ** 2 for a in U[i])); tw = mp.sqrt(sum(M(a) ** 2 for a in W[i]))
        Sb.append(nu_ * (tw * tu ** n) ** (2 / nu_))
    for lab, SS in [('as given', [M(x) for x in S]), ('balanced', Sb)]:
        w = [x / mp.fsum(SS) for x in SS]; H = -mp.fsum([x * mp.log(x) for x in w])
        print(f"   {lab}: cost shares min/max = {mp.nstr(min(w),6)}/{mp.nstr(max(w),6)}; H = {mp.nstr(H,8)}, log m = {mp.nstr(mp.log(m),8)}, log nu = {mp.nstr(mp.log(nu),8)}")
    # (c) integrated identity for the balanced network: log(gamma/gamma_IF)(s) at s=nu/2 vs crossing point
    hm = mp.mpf(h.numerator) / h.denominator
    def lr(s):
        b = nu / s
        return mp.log(hm / mp.fsum([x ** (b / 2) for x in Sb]) ** s / (nu ** (1 - s) * 2 * (6 * nu) ** (-mp.mpf(nu) / 2) * mp.mpf(1.5)))
    sc = mp.findroot(lr, (mp.mpf(1.0 + 0.5), mp.mpf(nu) / 2))
    wS = [x / mp.fsum(Sb) for x in Sb]; H = -mp.fsum([x * mp.log(x) for x in wS])
    print(f"   balanced: log ratio at b=2 = {mp.nstr(lr(mp.mpf(nu)/2),10)}; crossing s_c = {mp.nstr(sc,10)} (b={mp.nstr(nu/sc,8)}); (log nu - H(b=2))*(nu/2 - s_c) = {mp.nstr((mp.log(nu)-H)*(mp.mpf(nu)/2-sc),10)}")
    # (d) R13 single-neuron removal with gamma_cex as stand-in for gamma*
    s = mp.mpf(nu) / 2; P = mp.fsum([M(x) for x in S]); worst = None
    for i in range(m):
        rest = [k for k in range(m) if k != i]
        hR, argR = margin_counts(U, W, n, rest)
        gA = gap_at(U, W, [i], argR)
        PA = M(S[i])
        rhs = mp.mpf(gam.numerator) / gam.denominator * (P ** s - (P - PA) ** s)
        r = (mp.mpf(gA.numerator) / gA.denominator) / rhs
        worst = r if worst is None else min(worst, r)
    print(f"   R13 test (gamma* -> gamma_cex): min_i g_i(x_R,d_R)/(gamma (P^s-(P-P_i)^s)) = {mp.nstr(worst,6)} (>=1 means no single-neuron removal improves)")
