# DC-free (free-mode) relaxation for p=3: given structured Psi, max h s.t. exists Xi (free modes) with
#   Re(Psi+Xi)(x) - |Im(Psi+Xi)(x)|/sqrt3 >= h  for all x in Z_3^n
import numpy as np, itertools, sys
from scipy.optimize import linprog
w = np.exp(2j * np.pi / 3)
def hstar(Psi, n):
    X = np.array(list(itertools.product(range(3), repeat=n)))
    E = np.array(list(itertools.product(range(3), repeat=n)))
    free = [e for e in E if (e == 2).any()]   # characters w^{e.x} with some e_t=2 (= w^{-x_t}): free
    C = np.array([[w ** (e @ x) for e in free] for x in X])  # N x F complex
    N, F = C.shape
    # variables: [Re xi (F), Im xi (F), h]
    # Re(Psi + C xi) = Re Psi + Re C a - Im C b ; Im(...) = Im Psi + Im C a + Re C b
    s3 = 1 / np.sqrt(3)
    ReC, ImC = C.real, C.imag
    rows = []; rhs = []
    for sgn in (+1, -1):
        # Re - sgn*Im/sqrt3 >= h  ->  -(Re - sgn Im/s3) + h <= 0
        A_a = ReC - sgn * s3 * ImC
        A_b = -ImC - sgn * s3 * ReC
        base = Psi.real - sgn * s3 * Psi.imag
        rows.append(np.hstack([-A_a, -A_b, np.ones((N, 1))])); rhs.append(base)
    A = np.vstack(rows); b = np.concatenate(rhs)
    c = np.zeros(2 * F + 1); c[-1] = -1
    res = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] * (2 * F + 1), method="highs")
    return res.x[-1] if res.status == 0 else None
def psi_of(beta, xi, n):
    X = np.array(list(itertools.product(range(3), repeat=n)))
    return np.array([sum(b * np.prod(1 + xx * w ** x) for b, xx in zip(beta, xi)) for x in X])
if __name__ == "__main__":
    for n in range(1, 7):
        best = -1
        rng = np.random.default_rng(0)
        for trial in range(40):
            xi = [np.exp(1j * rng.uniform(0, 2 * np.pi, n))] if trial else [np.ones(n)]
            Psi = psi_of([1.0], xi, n)
            hs = hstar(Psi, n)
            best = max(best, hs)
        nu = n + 1
        print(f"n={n} single-neuron relaxed h*/|beta| = {best:.5f}   nu^(1-s) at b=2: {nu**(1-nu/2):.5f}", flush=True)
