# Independent referee check of largen R1 (width theorem for arbitrary sign patterns, slot-symmetric single features).
# (1) Verifies the key identity Phi(N(k)) = e^{i(pi-theta)k} S(k), S(k) = sum_i C_i r_i^k with r_i>0, computing Phi from the
#     logit table by the DEFINITION Phi(x) = (2/p) rho^{y} sum_c f(x)[c] rho^{-c} (high precision, mpmath).
# (2) On positive-margin networks (the certified p=3 counterexamples, the Note's p=5 1-D networks, and fresh random
#     positive-margin networks found by optimisation from random starts), checks for every b and k0 that the values
#     h(k) = Im(e^{-i psi} S(k)) at determined k alternate in sign, and that width >= K+1-floor((K+1)/p).
import json, math, itertools, sys
import numpy as np
import mpmath as mp
from fractions import Fraction as Fr
from scipy.optimize import minimize
mp.mp.dps = 50
rng = np.random.default_rng(777)
ROOT = '/home/user/grok/heracles/conjecture20/'

def counts(n, p):
    def rec(prefix, left, k):
        if k == p - 1:
            yield prefix + (left,); return
        for v in range(left + 1):
            yield from rec(prefix + (v,), left - v, k + 1)
    return list(rec((), n, 0))

def phi_at(U, W, N, p):
    # U: list of m input vectors (mp), W: list of m output vectors (mp); N a count vector.
    rho = mp.expj(2 * mp.pi / p)
    y = sum(a * N[a] for a in range(p)) % p
    f = [mp.fsum([W[i][c] * mp.fprod([U[i][a] ** N[a] for a in range(p)]) for i in range(len(U))]) for c in range(p)]
    return (mp.mpf(2) / p) * rho ** y * mp.fsum([f[c] * rho ** (-c) for c in range(p)])

def cmz(z, p):
    return mp.re(z) - abs(mp.im(z)) * mp.cot(mp.pi / p)

def margin_cm(U, W, p, n):
    return min(cmz(phi_at(U, W, N, p), p) for N in counts(n, p))

def check_alternation(U, W, p, n, label):
    m = len(U); th = mp.pi / p; K = n // 2
    d = (p - 1) // 2; dp = (d * pow(2, -1, p)) % p
    viol = 0; tests = 0; ident_err = mp.mpf(0)
    rho = mp.expj(2 * mp.pi / p)
    for b in range(p):
        a = (b + dp) % p
        if any(U[i][b] == 0 or U[i][a] == 0 for i in range(m)):
            continue
        Fk = []
        for k in range(K + 1):
            N = [0] * p; N[b] += n - 2 * k; N[a] += 2 * k
            Fk.append(phi_at(U, W, N, p))
        # identity check: C_i = (2/p) what_i(1) (u_i(b) rho^b)^n, r_i = (u_i(a)/u_i(b))^2
        what = [mp.fsum([W[i][c] * rho ** (-c) for c in range(p)]) for i in range(m)]
        C = [(mp.mpf(2) / p) * what[i] * (U[i][b] * rho ** b) ** n for i in range(m)]
        r = [(U[i][a] / U[i][b]) ** 2 for i in range(m)]
        S = [Fk[k] / mp.expj((mp.pi - th) * k) for k in range(K + 1)]
        for k in range(K + 1):
            ident_err = max(ident_err, abs(S[k] - mp.fsum([C[i] * r[i] ** k for i in range(m)])) / (1 + abs(S[k])))
        for k0 in range(min(p, K + 1)):
            psi = -(mp.pi - th) * k0
            h = [mp.im(mp.expj(-psi) * S[k]) for k in range(K + 1)]
            det = [k for k in range(K + 1) if (k - k0) % p != 0]
            sg = [mp.sign(h[k]) for k in det]
            tests += 1
            if any(sg[j] == 0 or sg[j] == sg[j + 1] for j in range(len(sg) - 1)) or (len(sg) == 1 and sg[0] == 0):
                viol += 1
    bound = K + 1 - (K + 1) // p
    print(f"  {label}: p={p} n={n} width={m} bound K+1-floor((K+1)/p)={bound} -> {'OK' if m >= bound else 'VIOLATED'};"
          f" alternation tests {tests}, violations {viol}; identity rel.err {mp.nstr(ident_err, 3)}", flush=True)

print("(a) certified p=3 counterexamples (exact_certificate.py weights)")
sys.path.insert(0, ROOT)
import importlib.util
spec = importlib.util.spec_from_file_location('excert', ROOT + 'exact_certificate.py'); ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)
for n, net in ex.NETWORKS.items():
    U = [[mp.mpf(Fr(v).numerator) / Fr(v).denominator for v in row] for row in net['u']]
    W = [[mp.mpf(Fr(v).numerator) / Fr(v).denominator for v in row] for row in net['w']]
    c = margin_cm(U, W, 3, n)
    print(f"  n={n}: min cm(Phi) = {mp.nstr(c, 8)} (positive margin: {c > 0})")
    check_alternation(U, W, 3, n, f"exact_certificate n={n}")

def oned_net(J):
    p, n = J['p'], J['n']
    lam, lv, av = J['lam'], J['logV'], J['argV']
    rho = mp.expj(2 * mp.pi / p)
    U, W = [], []
    for i in range(len(lam)):
        U.append([(1 if a < p / 2 else -1) * mp.e ** (mp.mpf(lam[i]) * (2 * a if a < p / 2 else 2 * a - p)) for a in range(p)])
        V = mp.e ** (mp.mpf(lv[i]) - max(lv)) * mp.expj(av[i])
        W.append([mp.re(V * rho ** c) for c in range(p)])
    return U, W

print("(b) certified 1-D p=3 networks (oned_certified) and the Note's p=5,7 1-D networks")
for fn in ['oned_certified/p3_n11.json', 'oned_certified/p3_n14.json', 'oned_certified/p3_n18.json',
           'mechanism_scripts/kkt/oned_p5_n6.json', 'mechanism_scripts/kkt/oned_p5_n4.json', 'mechanism_scripts/kkt/oned_p7_n3.json']:
    J = json.load(open(ROOT + fn)); U, W = oned_net(J)
    c = margin_cm(U, W, J['p'], J['n'])
    print(f"  {fn}: min cm = {mp.nstr(c, 6)}")
    if c > 0:
        check_alternation(U, W, J['p'], J['n'], fn)

print("(c) fresh positive-margin slot-symmetric single features with arbitrary inputs, found from random starts")
def fast_margin(z, m, p, n, Ns, ys):
    U = z[:m * p].reshape(m, p); Wc = z[m * p:m * p + m] * np.exp(1j * z[m * p + m:m * p + 2 * m])
    rho = np.exp(2j * np.pi * np.arange(p) / p)
    L = np.log((U * rho[None]).astype(complex))
    Phi = np.exp(Ns @ L.T) @ Wc
    val = (Phi.real - np.abs(Phi.imag) / math.tan(math.pi / p)).min()
    cost = np.sum(np.abs(Wc) * np.linalg.norm(U, axis=1) ** n)
    return val / cost

found = 0
for (p, n, m) in [(5, 6, 6), (5, 6, 7), (5, 8, 8), (7, 6, 7), (3, 6, 5), (3, 8, 6), (5, 9, 9)]:
    Ns = np.array(counts(n, p), float); ys = None
    for trial in range(30):
        z0 = np.concatenate([rng.standard_normal(m * p), np.abs(rng.standard_normal(m)), rng.uniform(0, 2 * np.pi, m)])
        res = minimize(lambda z: -fast_margin(z, m, p, n, Ns, ys), z0, method='Powell', options=dict(maxiter=4000))
        if -res.fun > 1e-9:
            z = res.x
            U = [[mp.mpf(float(v)) for v in z[i * p:(i + 1) * p]] for i in range(m)]
            Wc = z[m * p:m * p + m] * np.exp(1j * z[m * p + m:m * p + 2 * m])
            rho = mp.expj(2 * mp.pi / p)
            # output w_i(c) = Re(conj?) choose w_i(c) = Re(Wc_i rho^c) -> (2/p) what_i(1) = Wc_i
            W = [[mp.re(mp.mpc(Wc[i]) * rho ** c) for c in range(p)] for i in range(m)]
            c = margin_cm(U, W, p, n)
            if c > 0:
                found += 1
                signs = len(set(tuple(np.sign(z[i * p:(i + 1) * p])) for i in range(m)))
                print(f"   found positive margin (p={p},n={n},m={m}), distinct sign patterns among neurons: {signs}")
                check_alternation(U, W, p, n, f"random-start p={p} n={n}")
                break
    else:
        print(f"   (p={p},n={n},m={m}): no positive-margin network found in 30 random starts")
print("done; random positive-margin networks checked:", found)
