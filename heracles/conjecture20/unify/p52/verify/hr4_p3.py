"""hr4: test R12 (delta-Fourier isolation) and the single-feature trivial bound against the p=3 counterexamples
(exact_certificate.py networks, n=11 and n=13, b=2), in mpmath high precision via count vectors."""
import sys
import mpmath as mp
sys.path.insert(0, '/home/user/grok/heracles/conjecture20')
import importlib.util
spec = importlib.util.spec_from_file_location('ec', '/home/user/grok/heracles/conjecture20/exact_certificate.py')
ec = importlib.util.module_from_spec(spec); spec.loader.exec_module(ec)
mp.mp.dps = 40
p = 3
rho = mp.exp(2j * mp.pi / p)
for n, net in ec.NETWORKS.items():
    nu = n + 1
    U = [[mp.mpf(a) for a in row] for row in net["u"]]
    W = [[mp.mpf(a) for a in row] for row in net["w"]]
    m = len(U)
    # margin over count vectors
    h = None
    for N0 in range(n + 1):
        for N1 in range(n + 1 - N0):
            N2 = n - N0 - N1
            y = (N1 + 2 * N2) % p
            z = [U[i][0] ** N0 * U[i][1] ** N1 * U[i][2] ** N2 for i in range(m)]
            f = [sum(W[i][c] * z[i] for i in range(m)) for c in range(p)]
            for c in range(p):
                if c != y:
                    g = f[y] - f[c]
                    h = g if h is None or g < h else h
    P2 = sum(n * sum(a * a for a in U[i]) + sum(a * a for a in W[i]) for i in range(m))
    s = mp.mpf(nu) / 2
    hat = lambda v, k: sum(v[a] * rho ** (-k * a) for a in range(p))
    beta = sum(2 * mp.mpf(p) ** (-nu) * mp.conj(hat(W[i], 1)) * hat(U[i], 1) ** n for i in range(m))
    bT = [abs(sum(2 * mp.mpf(p) ** (-nu) * mp.conj(hat(W[i], 1)) * hat(U[i], -1) ** k * hat(U[i], 1) ** (n - k)
                  for i in range(m))) for k in range(1, n + 1)]
    Bp = 2 * (2 * mp.mpf(p) * nu) ** (-mp.mpf(nu) / 2)
    gamma = h / P2 ** s
    gIF = nu ** (1 - s) * Bp * mp.mpf(3) / 2
    r12 = [mp.mpf(3) / 2 * mp.re(beta) - mp.mpf(3) / 4 * x for x in bT]
    lem2 = mp.mpf(3) / 2 * mp.re(beta)
    print(f"n={n}: gamma/gamma_IF = {mp.nstr(gamma/gIF, 8)};  h = {mp.nstr(h, 10)};  Lemma2 bound 1.5 Re beta = {mp.nstr(lem2, 10)}")
    print(f"   R12 bounds (|T| = 1..n): min = {mp.nstr(min(r12), 10)}, all >= h: {all(h <= v for v in r12)};"
          f"  max_T |beta_T| / |beta| = {mp.nstr(max(bT)/abs(beta), 6)}")
