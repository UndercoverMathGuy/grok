"""Math referee: short spot-check of the (5,2) certificate step (R9) with MY OWN degree-2 Hermitian-square SDP
(cvxpy, float, not rigorous) at a few normal-form own-data points, with my own payment F (grid, mr3_payment.tw_pay).
Checks that sqrt(a_+^2/F^2 + 12 n^2)/Gamma_{3/2} <= 0.995 is attainable where the trivial bound fails."""
import itertools
import numpy as np
import cvxpy as cp
from mr3_payment import tw_pay, plb

p = 5; rho = np.exp(2j * np.pi / p)
E = list(itertools.product(range(3), repeat=2))
XS = np.array(list(itertools.product(range(p), repeat=2)))
KA = 1 - np.cos(2 * np.pi / 5); GAM = 2.5 * 2 ** -1.5
c = {j: np.array([1 - rho ** (-j * d) for d in (1, 2, 3, 4)]) for j in (1, 2)}

# linear map: G_delta (9x9) -> q(x,delta) = v^* G v / 25
Vx = []
for x in XS:
    th = 2 * np.pi * 2 * x / 5
    v = np.exp(1j * np.array([e[0] * th[0] + e[1] * th[1] for e in E]))
    Vx.append(np.outer(np.conj(v), v))       # q = sum_{e,e'} conj(v_e) G[e,e'] v_e' = sum (conj(v) v^T) .* G


def solve(alpha, m1, m2, m12, phpay, F):
    A = np.exp(1j * alpha); B = m1; C = m2; D = m12 * np.exp(1j * (phpay - alpha))
    Gs = [cp.Variable((9, 9), hermitian=True) for _ in range(4)]
    cons = [G >> 0 for G in Gs] + [sum(cp.real(cp.trace(G)) for G in Gs) == 1]
    q = [[cp.real(cp.sum(cp.multiply(Vx[xi], Gs[di]))) / 25 for di in range(4)] for xi in range(25)]
    def chat25(j, psi):     # 25 * Chat_j(psi) = sum_x sum_d q c_j(d) rho^{-<psi,x>}
        terms = []
        for xi, x in enumerate(XS):
            ph = rho ** (-(x @ np.array(psi)))
            for di in range(4):
                terms.append(q[xi][di] * (c[j][di] * ph))
        return cp.sum(cp.hstack(terms))
    a0 = chat25(1, (0, 0))
    aT = [chat25(1, tuple((-np.array(chi)) % p)) for chi in [(2, 0), (0, 2), (2, 2)]]
    a = cp.real(np.conj(A) * a0) + cp.real(np.conj(B) * aT[0]) + cp.real(np.conj(C) * aT[1]) + cp.real(np.conj(D) * aT[2])
    nv = []
    for j in (1, 2):
        for chi in itertools.product(range(p), repeat=2):
            if (j == 1 and chi in [(0, 0), (2, 0), (0, 2), (2, 2)]) or (j == 2 and chi == (0, 0)):
                continue
            z = chat25(j, tuple((-np.array(chi)) % p))
            nv += [cp.real(z), cp.imag(z)]
    obj = cp.norm(cp.hstack([a / F] + [np.sqrt(12) * t for t in nv]), 2)
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=cp.CLARABEL)
    # recompute bound with a_+ from the solution
    av = a.value; nn = np.sqrt(sum(t.value ** 2 for t in nv))
    return np.sqrt(max(av, 0) ** 2 / F ** 2 + 12 * nn ** 2) / GAM, prob.status


if __name__ == '__main__':
    rng = np.random.default_rng(11)
    pts = [(0.0, 1.0, 1.0, 1.0, 0.0), (0.3, 1.0, 1.0, 1.0, 0.0), (0.0, 0.8, 0.5, 0.6, 2.0)]
    for _ in range(4):
        m = np.sort(rng.uniform(0.2, 1.4, 2))[::-1]
        pts.append((rng.uniform(0, 0.6), m[0], m[1], rng.uniform(0.2, 1.4), rng.uniform(-np.pi, np.pi)))
    for (al, m1, m2, m12, ph) in pts:
        F = max(tw_pay(1, m1, m2, m12, np.cos(ph), 1.5, nL=60, npsi=121),
                plb(1, max(m1, m2, m12), np.sqrt(m1 ** 2 + m2 ** 2 + m12 ** 2), 1.5))
        triv = KA * np.cos(al) / (GAM * F)
        val, st = solve(al, m1, m2, m12, ph, F)
        print(f"alpha={al:.3f} m=({m1:.3f},{m2:.3f},{m12:.3f}) phi_pay={ph:.3f}: F={F:.4f} trivial={triv:.4f}  my SDP cert={val:.4f} [{st}]"
              f"  -> min = {min(triv, val):.4f} (<= 0.995)")
