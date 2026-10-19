"""Phase-blind SEMI-RIGID channel-resolved bound at (p,n) = (7,2), features L = {1,2} (beta_3 = 0).

Test: q(x,d) = p^-2 [ mu_d + 2 Re(z1(d) rho^{-<chi1,x>}) + 2 Re(z2(d) rho^{-<chi2,x>}) ],
  chi1 = chi_{1,T1} (largest |beta_{1,T}|), chi2 = chi_{2,T2} (largest |beta_{2,T}|).
Positivity (phase-blind): mu_d >= 2|z1(d)| + 2|z2(d)|.
Pairing: sum q g = sum_l K_l(mu) Re beta_l - |S1| M1 - |S2| M2 + leak,  S_j = sum_d z_j(d) c_j(d),
  leak = sum over non-own (l, +-chi_j) of field coefficient * L,  all PAID except
  (l in {1,2}, -chi2): free  ->  constraint sum_d z2(d) conj(c_l(d)) = 0 (l = 1, 2).
|leak| <= R n,  n = || L_paid ||_2.   Cost ledger: 3^{1-s} B P^s >= sqrt(F^2 + c R^2), c = 3^{2-2s}/4.
=> gamma/gamma_IF <= sqrt(a_+^2/F^2 + n^2/c)/Gamma_s.
"""
import numpy as np
import cvxpy as cp

p = 7
rho = np.exp(2j * np.pi / p)
DEL = np.arange(1, p)
FF = [1, 2, 3]


def cf(l, d):
    return 1 - rho ** (-l * d)


def kap(l, d):
    return 1 - np.cos(2 * np.pi * l * d / p)


CM = np.array([[cf(l, d) for d in DEL] for l in FF])          # 3 x 6
KM = np.array([[kap(l, d) for d in DEL] for l in FF])         # 3 x 6


def Gamma7(s):
    return 3.5 * 3.0 ** (-s)


def payment_simple(x, s):
    c1 = 1 - 3.0 ** (1 - s)
    return max(1 - c1 * x, 3.0 ** (1 - s) * max(1.0, x))


class SemiRigid:
    def __init__(self, s, sym=True, use2=True):
        self.s = s
        self.c = 3.0 ** (2 - 2 * s) / 4
        self.mu = cp.Variable(6, nonneg=True)
        self.z1r = cp.Variable(6); self.z1i = cp.Variable(6)
        self.z2r = cp.Variable(6); self.z2i = cp.Variable(6)
        self.A = cp.Parameter(2, nonneg=True)     # A1, A2  (scaled by 1/F)
        self.M = cp.Parameter(2, nonneg=True)     # M1, M2  (scaled by 1/F)
        t = cp.Variable(nonneg=True)
        cons = [cp.sum(self.mu) == 1]
        if sym:
            cons += [self.mu[k] == self.mu[5 - k] for k in range(3)]
        for d in range(6):
            cons.append(self.mu[d] >= 2 * cp.norm(cp.hstack([self.z1r[d], self.z1i[d]]), 2)
                        + 2 * cp.norm(cp.hstack([self.z2r[d], self.z2i[d]]), 2))
        def lin(zr, zi, vec):   # sum_d z(d) vec(d) -> (re, im)
            return (zr @ vec.real - zi @ vec.imag, zr @ vec.imag + zi @ vec.real)
        # own detections (take real part as the detected amount; imag part free)
        S1 = lin(self.z1r, self.z1i, CM[0])[0]
        S2 = lin(self.z2r, self.z2i, CM[1])[0]
        if not use2:
            cons += [self.z2r == 0, self.z2i == 0]
        # free constraints for component 2
        for l in (0, 1):
            re, im = lin(self.z2r, self.z2i, np.conj(CM[l]))
            cons += [re == 0, im == 0]
        leaks = []
        # component 1 at chi1: l = 2,3 (paid);  at -chi1: l = 1,2,3 (paid)
        for l in (1, 2):
            leaks += list(lin(self.z1r, self.z1i, CM[l]))
        for l in (0, 1, 2):
            leaks += list(lin(self.z1r, self.z1i, np.conj(CM[l])))
        # component 2 at chi2: l = 1,3 (paid); at -chi2: l = 3 (paid); l=1,2 free (constrained 0)
        for l in (0, 2):
            leaks += list(lin(self.z2r, self.z2i, CM[l]))
        leaks += list(lin(self.z2r, self.z2i, np.conj(CM[2])))
        n = cp.norm(cp.hstack(leaks), 2)
        Kmu = KM @ self.mu
        a = self.A[0] * Kmu[0] + self.A[1] * Kmu[1] - S1 * self.M[0] - S2 * self.M[1]
        cons.append(t >= a)
        self.n = n; self.aexpr = a
        self.prob = cp.Problem(cp.Minimize(cp.norm(cp.hstack([t, n / np.sqrt(self.c)]), 2)), cons)

    def __call__(self, A1, A2, M1, M2, F):
        self.A.value = np.array([A1, A2]) / F
        self.M.value = np.array([M1, M2]) / F
        try:
            self.prob.solve(solver=cp.CLARABEL)
        except Exception:
            self.prob.solve(solver=cp.SCS, eps=1e-8)
        return self.prob.value / Gamma7(self.s)


def bound(sr, tau, x1, x2, pay=payment_simple):
    s = sr.s; r = 1 / s
    A1, A2 = 1 - tau, tau
    F1 = pay(x1, s) if A1 > 0 else 0.0
    F2 = pay(x2, s) if A2 > 0 else 0.0
    F = ((F1 * A1) ** r + (F2 * A2) ** r) ** s
    return sr(A1, A2, x1 * A1, x2 * A2, F)


if __name__ == '__main__':
    import sys, time
    s = float(sys.argv[1]) if len(sys.argv) > 1 else 1.2
    sr = SemiRigid(s)
    tau0 = 2 * np.cos(np.pi / 7) / (1 + 2 * np.cos(np.pi / 7))
    t0 = time.time()
    print(f"s={s}: IF tie point bound {bound(sr, tau0, 0, 0):.5f}  [{time.time()-t0:.2f}s]")
    best = (0, None)
    for tau in np.linspace(0.0, 1.0, 21):
        for x1 in np.linspace(0, 2.0, 21):
            for x2 in np.linspace(0, 2.0, 21):
                v = bound(sr, tau, x1, x2)
                if v > best[0]:
                    best = (v, (tau, x1, x2))
    print(f"grid sup {best[0]:.5f} at tau,x1,x2 = {best[1]}  [{time.time()-t0:.1f}s]")
