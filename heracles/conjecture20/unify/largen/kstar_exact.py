# Exact (mpmath) evaluation of slot-symmetric atom systems for k*(n,s).
#   atoms beta_i, phases xi_i = e^{i phi_i}; m_k = sum_i beta_i xi_i^k; E = sum_{k=1}^n C(n,k)|m_k|^2 = beta^H G beta,
#   G_ij = (1 + conj(xi_i) xi_j)^n - 1.   IF-free optimum over beta: beta* = G^{-1} 1, ratio sqrt(2 * 1^T G^{-1} 1).
#   Full ratio with the IF price: (Re A - nu^{1-s} (sum |beta_i|^{1/s})^s) / sqrt(E/2).
# Positions: equally spaced arc phi_i = Delta (i - (m-1)/2).  We scan (m, Delta), take beta* and then improve beta by
# a few Newton-free gradient steps in mpmath (the IF price is a small correction at large n).
import sys, mpmath as mp

def G_matrix(phis, n):
    m = len(phis)
    xi = [mp.expjpi(ph / mp.pi) for ph in phis]
    G = mp.matrix(m, m)
    for i in range(m):
        for j in range(m):
            G[i, j] = (1 + mp.conj(xi[i]) * xi[j]) ** n - 1
    return G, xi

def ratio_full(beta, G, n, s):
    nu = n + 1
    A = mp.fsum(beta)
    E = mp.re((mp.matrix([[mp.conj(b) for b in beta]]) * G * mp.matrix(beta))[0, 0])
    Q = mp.fsum([abs(b) ** (1 / mp.mpf(s)) for b in beta])
    IF = mp.mpf(nu) ** (1 - mp.mpf(s)) * Q ** mp.mpf(s)
    return (mp.re(A) - IF) / mp.sqrt(E / 2), A, E, IF

def eval_arc(n, m, Delta, s, dps=60):
    mp.mp.dps = dps
    phis = [mp.mpf(Delta) * (i - mp.mpf(m - 1) / 2) for i in range(m)]
    G, xi = G_matrix(phis, n)
    one = mp.matrix([1] * m)
    x = mp.lu_solve(G, one)
    Xi = mp.re(mp.fsum([x[i] for i in range(m)]))
    beta = [x[i] / Xi for i in range(m)]              # normalised so that A = 1
    r, A, E, IF = ratio_full(beta, G, n, s)
    L = mp.fsum([abs(b) for b in beta])
    return dict(ratio_IFfree=mp.sqrt(2 * Xi), ratio=r, A=A, E=E, IF=IF, L=L, beta=beta, G=G, phis=phis)

if __name__ == "__main__":
    bval = 2.0
    for n in [int(v) for v in sys.argv[1].split(',')]:
        s = (n + 1) / bval
        best = None
        for frac in [0.66, 0.70, 0.74, 0.78, 0.82]:
            m = max(2, int(round(frac * n)))
            for c in [1.30, 1.38, 1.46, 1.54, 1.62, 1.70]:
                Delta = 2 * float(mp.pi) / (c * n + 2)
                d = eval_arc(n, m, Delta, s, dps=40 + n)
                if best is None or d['ratio'] > best[0]['ratio']:
                    best = (d, m, c)
        d, m, c = best
        print(f"n={n}: arc m={m} Delta=2pi/({c}n+2): ratio(with IF price, beta=G^-1 1) = {mp.nstr(d['ratio'],6)}"
              f"  IF-free sqrt(2Xi) = {mp.nstr(d['ratio_IFfree'],6)}  IF/A = {mp.nstr(d['IF'],4)}  L/A = {mp.nstr(d['L'],5)}"
              f"  E/A^2 = {mp.nstr(d['E'],5)}", flush=True)
