"""Math referee: independent checks of R7 (twisted Rotfel'd payment, (P,q)-monotonicity, S_min formula, disk-to-circle
reduction) and R8 (s-monotonicity of every payment bound).  Own implementation from the statement in REPORT_R2."""
import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(7)


def lowK(P):
    return np.maximum(P - 1, -2 * (P / 3) ** 1.5)


def upK(P):
    return 2 * (P / 3) ** 1.5


def roots(P, q):
    m = 2 * np.sqrt(max(P, 0) / 3)
    if m == 0:
        return np.zeros(3)
    arg = np.clip(3 * q / (P * m), -1, 1)
    th = np.arccos(arg) / 3
    return np.sort(m * np.cos(th - 2 * np.pi * np.arange(3) / 3))


def S(P, q, r):
    mu = roots(P, q)
    return np.sum(np.maximum(1 + mu, 0) ** r)


def Smin(Pbar, qlo, r):
    Pc = min(max(Pbar, 0.0), 3.0)
    if qlo > upK(Pc) + 1e-15:
        return np.inf
    return S(Pc, max(qlo, lowK(Pc)), r)


def atoms(k, mode='rand'):
    if mode == 'rand':
        vr = rng.exponential(size=k) ** rng.uniform(0.3, 3)
        pp = np.exp(1j * rng.uniform(-np.pi, np.pi, k) * rng.uniform(0, 1))
        xi = np.exp(1j * rng.uniform(-np.pi, np.pi, k)); eta = np.exp(1j * rng.uniform(-np.pi, np.pi, k))
    elif mode == 'nearIF':
        vr = np.ones(k) + 0.1 * rng.random(k); pp = np.exp(1j * 0.2 * rng.normal(size=k))
        g = np.exp(2j * np.pi * np.arange(k) / k)
        xi = g * np.exp(0.3j * rng.normal(size=k)); eta = g * np.exp(0.3j * rng.normal(size=k))
    else:   # near single atom with small companions
        vr = np.concatenate([[1.0], 0.2 * rng.random(k - 1)]); pp = np.exp(1j * rng.normal(size=k) * 0.5)
        xi = np.exp(1j * rng.uniform(-np.pi, np.pi, k)); eta = np.exp(1j * rng.uniform(-np.pi, np.pi, k))
    return vr, pp, xi, eta


def own(vr, pp, xi, eta):
    b = vr * pp
    return b.sum(), (b * xi).sum(), (b * eta).sum(), (b * xi * eta).sum()


def chains(vr, pp, xi, eta):
    out = []
    for c in range(3):
        if c == 0:
            u = np.stack([np.ones_like(pp), pp * xi, pp ** 2 * xi * eta], 1)
        elif c == 1:
            u = np.stack([np.ones_like(pp), pp * xi, pp * xi * eta], 1)
        else:
            u = np.stack([np.ones_like(pp), pp * eta, pp * xi * eta], 1)
        H = np.einsum('i,ia,ib->ab', vr, u, u.conj())
        out.append(H)
    return out


def test_entries_rotfeld():
    worst_rot = -np.inf; worst_ent = 0; worst_disk = -np.inf
    for t in range(3000):
        k = rng.integers(1, 7)
        vr, pp, xi, eta = atoms(k, ['rand', 'nearIF', 'single'][t % 3])
        A, B, C, D = own(vr, pp, xi, eta)
        L = vr.sum(); al = A / abs(A); rhoL = np.sqrt(max(2 * L * (L - abs(A)), 0))
        H1, H2, H3 = chains(vr, pp, xi, eta)
        # exact entries and uncertain entries (H[k,l] = sum vr u_k conj(u_l))
        worst_ent = max(worst_ent, abs(H1[1, 0] - B), abs(H1[2, 1] - C), abs(H2[1, 0] - B), abs(H2[2, 0] - D),
                        abs(H3[1, 0] - C), abs(H3[2, 0] - D), *[abs(np.diag(H) - L).max() for H in (H1, H2, H3)])
        worst_disk = max(worst_disk, abs(H1[2, 0] - al * D) - rhoL, abs(H2[2, 1] - np.conj(al) * C) - rhoL,
                         abs(H3[2, 1] - np.conj(al) * B) - rhoL)
        for s in (1.0, 1.25, 1.5):
            r = 1 / s
            P1 = 3 ** (1 - s) * np.sum(vr ** r) ** s
            for H in (H1, H2, H3):
                nuv = np.clip(np.linalg.eigvalsh(H / L), 0, None)
                bound = L * (np.mean(nuv ** r)) ** s
                worst_rot = max(worst_rot, (bound - P1) / P1)
    print(f"R7(a) entries: max err {worst_ent:.2e};  max[|X'-alpha^(+-1)X| - rho_L] = {worst_disk:.2e} (<=0);"
          f"  Rotfeld max rel[L M_r(nu) - P_1] = {worst_rot:.2e} (<=0)")


def test_monotone():
    viol = 0; tot = 0
    for t in range(20000):
        r = rng.uniform(2 / 3, 1)
        P = rng.uniform(0, 3); lo, hi = lowK(P), upK(P)
        if hi <= lo:
            continue
        q = rng.uniform(lo, hi)
        dq = rng.uniform(0, hi - q) * rng.random()
        dP = rng.uniform(0, 3 - P) * rng.random() * 0.2
        s0 = S(P, q, r)
        if q + dq <= upK(P):
            tot += 1; viol += S(P, q + dq, r) < s0 - 1e-12
        if q >= lowK(P + dP) and q <= upK(P + dP):
            tot += 1; viol += S(P + dP, q, r) > s0 + 1e-12
    print(f"R7(b) monotonicity (S up in q, down in P): violations {viol}/{tot}")


def test_smin():
    worst = 0
    for t in range(400):
        r = rng.choice([2 / 3, 0.8, 1.0])
        Pbar = rng.uniform(0, 3.5); qlo = rng.uniform(-2.2, 2.2)
        f = Smin(Pbar, qlo, r)
        # brute force over feasible grid
        best = np.inf
        for P in np.linspace(0, min(Pbar, 3), 300):
            lo, hi = max(lowK(P), qlo), upK(P)
            if lo > hi:
                continue
            for q in np.linspace(lo, hi, 40):
                best = min(best, S(P, q, r))
        if np.isinf(f) and np.isinf(best):
            continue
        worst = max(worst, abs(f - best) if np.isfinite(best) else (0 if np.isinf(f) else 99))
        if f > best + 1e-9:
            print("   S_min formula ABOVE brute force:", Pbar, qlo, r, f, best)
    print(f"R7(c) S_min closed form vs brute-force grid min: max |diff| = {worst:.2e} (grid resolution)")


def test_circle():
    bad = 0; tot = 0
    for t in range(3000):
        mX, mY, mZ = rng.uniform(0, 1.5, 3); ph = rng.uniform(-np.pi, np.pi)
        L = rng.uniform(max(1, mX, mY, mZ), 3); rho = rng.uniform(0, 1.5)
        Y = mY * np.exp(1j * rng.uniform(-np.pi, np.pi)); Z = mZ * np.exp(1j * rng.uniform(-np.pi, np.pi))
        c0 = Y * Z / abs(Y * Z) * np.exp(-1j * ph) * mX if mY * mZ > 0 else mX
        # X' interior
        Xp = c0 + rho * np.sqrt(rng.random()) * np.exp(1j * rng.uniform(-np.pi, np.pi))
        P = (abs(Xp) ** 2 + mY ** 2 + mZ ** 2) / L ** 2; q = 2 * (Y * Z * np.conj(Xp)).real / L ** 3
        psi = np.linspace(-np.pi, np.pi, 20001)
        Pb = (mY ** 2 + mZ ** 2 + mX ** 2 + rho ** 2 + 2 * mX * rho * np.cos(psi - ph)) / L ** 2
        qc = 2 * mY * mZ * (mX * np.cos(ph) + rho * np.cos(psi)) / L ** 3
        ok = np.any((Pb >= P - 1e-6) & (qc <= q + 1e-6))
        tot += 1; bad += (not ok)
    print(f"R7(d) disk->circle: interior points with no dominating circle point: {bad}/{tot}")


def plb(A, M, RT, s):
    c1 = 1 - 3 ** (1 - s)
    return max(A - c1 * M, A - RT / (2 * np.sqrt(3)), 3 ** (1 - s) * max(A, M))


def tw_pay(mA, mB, mC, mD, cph, s, nL=120, npsi=181):
    """grid version (not rigorous) of the circle-coupled 3-chain twisted payment"""
    r = 1 / s
    Lmin = max(mA, mB, mC, mD)
    ph = np.arccos(np.clip(cph, -1, 1))
    best_chain = -np.inf
    for (mX, mY, mZ) in ((mD, mB, mC), (mC, mB, mD), (mB, mC, mD)):
        val = 3 ** (1 - s) * 4 * Lmin            # tail L >= 4 Lmin
        for L in Lmin * (1 + 3 * np.linspace(0, 1, nL) ** 2):
            rho = np.sqrt(max(2 * L * (L - mA), 0))
            for psi in np.linspace(-np.pi, np.pi, npsi):
                for phs in (ph, -ph):
                    Pb = (mY ** 2 + mZ ** 2 + mX ** 2 + rho ** 2 + 2 * mX * rho * np.cos(psi - phs)) / L ** 2
                    qq = 2 * mY * mZ * (mX * np.cos(phs) + rho * np.cos(psi)) / L ** 3
                    sm = Smin(Pb, qq, r)
                    if np.isfinite(sm):
                        val = min(val, L * (sm / 3) ** s)
        best_chain = max(best_chain, val)
    return best_chain


def test_full_payment():
    worst = np.inf; rec = None
    for t in range(60):
        k = rng.integers(1, 6)
        vr, pp, xi, eta = atoms(k, ['rand', 'nearIF', 'single'][t % 3])
        A, B, C, D = own(vr, pp, xi, eta)
        sc = abs(A)
        if sc < 1e-6:
            continue
        mA, mB, mC, mD = 1.0, abs(B) / sc, abs(C) / sc, abs(D) / sc
        cph = np.cos(np.angle(A * D * np.conj(B * C)))
        for s in (1.0, 1.25, 1.5):
            P1 = 3 ** (1 - s) * np.sum((vr / sc) ** (1 / s)) ** s
            F = max(tw_pay(mA, mB, mC, mD, cph, s, nL=40, npsi=61), plb(1, max(mB, mC, mD), np.sqrt(mB ** 2 + mC ** 2 + mD ** 2), s))
            if P1 - F < worst:
                worst = P1 - F; rec = (k, s, mB, mC, mD, cph, P1, F)
    print(f"R7    full payment (grid): min[P_1 - F] = {worst:.4f} (>= -grid error);  worst case {np.round(rec,4)}")


def test_adversarial_payment():
    """minimise P_1 - F over 2..3-atom systems (F computed on a coarse grid -> F slightly overestimated)"""
    worst = np.inf
    for st in range(6):
        k = 2 + st % 2
        s = (1.5, 1.2)[st % 2]

        def obj(v):
            vr = np.exp(v[:k]); pp = np.exp(1j * v[k:2 * k]); xi = np.exp(1j * v[2 * k:3 * k]); eta = np.exp(1j * v[3 * k:4 * k])
            A, B, C, D = own(vr, pp, xi, eta)
            sc = abs(A)
            if sc < 1e-3 * vr.sum():
                return 1.0
            mB, mC, mD = abs(B) / sc, abs(C) / sc, abs(D) / sc
            cph = np.cos(np.angle(A * D * np.conj(B * C)))
            P1 = 3 ** (1 - s) * np.sum((vr / sc) ** (1 / s)) ** s
            F = max(tw_pay(1, mB, mC, mD, cph, s, nL=25, npsi=37), plb(1, max(mB, mC, mD), np.sqrt(mB ** 2 + mC ** 2 + mD ** 2), s))
            return P1 - F
        r = minimize(obj, rng.normal(size=4 * k), method='Nelder-Mead', options=dict(maxiter=400))
        worst = min(worst, r.fun)
    print(f"R7    adversarial min[P_1 - F] over 2-3 atoms (coarse grid F) = {worst:.4f}")


def test_s_monotone():
    viol = 0; tot = 0
    for t in range(40):
        mB, mC, mD = rng.uniform(0, 1.4, 3); cph = rng.uniform(-1, 1)
        Fs = [max(tw_pay(1, mB, mC, mD, cph, s, nL=30, npsi=61), plb(1, max(mB, mC, mD), np.sqrt(mB ** 2 + mC ** 2 + mD ** 2), s))
              for s in (1.0, 1.1, 1.25, 1.4, 1.5)]
        for a, b in zip(Fs[:-1], Fs[1:]):
            tot += 1; viol += b > a + 1e-9
    G = [2.5 * 2 ** -s for s in (1, 1.25, 1.5)]; c = [3 ** (2 - 2 * s) / 4 for s in (1, 1.25, 1.5)]
    print(f"R8    payment F_s non-increasing in s: violations {viol}/{tot};  Gamma_s {np.round(G,4)} and c_s {np.round(c,4)} decreasing")


if __name__ == '__main__':
    test_entries_rotfeld()
    test_monotone()
    test_smin()
    test_circle()
    test_s_monotone()
    test_full_payment()
    test_adversarial_payment()
