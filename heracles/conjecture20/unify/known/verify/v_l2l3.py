"""Referee checks for L2 (coverage-free O1 is false: intervals + Rudin-Shapiro signs) and L3 (R8a family).

L2: sigma_delta(S) (Note, Prop effsidon) = inf_{b>=0 on S, phases} (1/2) osc_w(sum b_l cos(2 pi l w/p + psi_l)) / sum_l b_l s_l(delta),
    s_l(delta) = |sin(pi l delta / p)|.  Take S = {1..K}, b_l = 1, psi_l in {0, pi} from the Rudin-Shapiro sequence.
    Upper bound on sigma: (1/2) osc / sum_l s_l(delta), with osc over w in Z_p (exact grid) and over real w (fine grid).
L3: n atoms beta_i = A/n, xi_it = e^{2 pi i i/n}: check P(z) = A(1 + z_1...z_n) (all beta_T), gain and payment constants.
    Plus a Bohr-type illustration for MULTI-AFFINE degree-2 polynomials: sum|c_T| / sup_{T^n}|H_2| for random signs.
"""
import itertools
import numpy as np
rng = np.random.default_rng(5)


def rudin_shapiro(K):
    e = []
    for l in range(K):
        bits = bin(l)[2:]
        cnt = sum(1 for i in range(len(bits) - 1) if bits[i] == '1' and bits[i + 1] == '1')
        e.append((-1) ** cnt)
    return np.array(e)


print("L2: intervals S={1..K} with Rudin-Shapiro signs")
p = 1000003
for K in [8, 32, 128, 512, 2048]:
    eps = rudin_shapiro(K)
    ls = np.arange(1, K + 1)
    w = np.arange(p)
    # exact grid w in Z_p via FFT of the coefficient vector
    coef = np.zeros(p, dtype=complex); coef[ls] = eps
    vals = np.real(np.fft.fft(coef))       # sum_l eps_l e^{-2 pi i l w/p}; real part = sum eps cos(2 pi l w/p)
    osc = vals.max() - vals.min()
    supabs = np.max(np.abs(np.fft.fft(coef)))
    delta = int(round(p / (2 * K)))
    den = np.sum(np.abs(np.sin(np.pi * ls * delta / p)))
    print(f"  K={K}: sup|RS poly| = {supabs:.2f} <= sqrt(2K) = {np.sqrt(2 * K):.2f}; (1/2)osc = {osc / 2:.2f}; "
          f"sum s_l(delta={delta}) = {den:.1f} = {den / K:.3f} K; sigma_delta <= {osc / 2 / den:.4f}")

print("L3: R8a family")
for n in [2, 3, 4, 5, 6, 7]:
    A = 1.0
    zeta = np.exp(2j * np.pi * np.arange(n) / n)
    worst = 0
    for k in range(0, n + 1):
        for T in itertools.combinations(range(n), k):
            bT = np.sum(A / n * zeta ** len(T))
            target = A if k in (0, n) else 0
            worst = max(worst, abs(bT - target))
    print(f"  n={n}: max |beta_T - target| = {worst:.1e}  (P = A(1 + z_1...z_n))")
for n in [4, 8, 16, 32, 64]:
    nu = n + 1; s = nu / 2
    gain = 1 - (n / nu) ** (s - 1)
    print(f"  b=2 n={n}: gain/A = {gain:.4f}; required C >= gain / R^n at R=1/2: {gain * 2 ** n:.4g}")

print("Bohr-type illustration, multi-affine degree 2: ratio sum|c_T| / sup_{T^n} |sum c_T z^T| (random signs)")
for n in [4, 8, 16, 32]:
    pairs = list(itertools.combinations(range(n), 2))
    rat = []
    for _ in range(3):
        c = rng.choice([-1.0, 1.0], size=len(pairs))
        Cm = np.zeros((n, n))
        for (a, b_), v in zip(pairs, c):
            Cm[a, b_] = Cm[b_, a] = v / 2
        best = 0
        for _r in range(300):
            th = rng.uniform(0, 2 * np.pi, n)
            for it in range(60):        # coordinate ascent on |z^T C z|
                z = np.exp(1j * th)
                val = z @ Cm @ z
                for t in range(n):
                    z = np.exp(1j * th)
                    g = 2 * (Cm[t] @ z) - 2 * Cm[t, t] * z[t]   # coefficient of z_t
                    rest = val - z[t] * g
                    # maximise |rest + e^{i a} g| over a: align phases
                    th[t] = np.angle(rest) - np.angle(g) if abs(rest) > 0 else th[t]
                    z = np.exp(1j * th); val = z @ Cm @ z
            best = max(best, abs(val))
        rat.append(len(pairs) / best)
    print(f"  n={n}: sum|c|/sup ~ {np.mean(rat):.2f} (sqrt(n) = {np.sqrt(n):.2f})")
