# Checks of the sharp plane Bhatia-Davis theorem (PBD#) and of every step of its proof.
#
#  f1,f2,f3 : Z_p -> R, mean zero;  F(x) = f1(x1)+f2(x2)+f3(x3);  Pi_c = {x1+x2+x3 = c}.
#  m_c = -min_{Pi_c} F,  M_c = max_{Pi_c} F,  V_i = (1/p) sum f_i^2,  V = sum V_i.
#  THEOREM:  m_c * M_c'  >=  V - (2/(p+1)) sum_{i<j} sqrt(V_i V_j)  >=  (p-1)/(p+1) V   for all c, c'.
#  Equality in the last form: f_i(x) = (-x mod p) - (p-1)/2 for i=1,2,3, (c,c') = (1,2).
#
#  Proof steps checked here:
#   (S1) Cauchy-Davenport step: with A_i = decreasing rearrangement of f_i and H(y) = sum_i A_i(y_i),
#        p-1 <= sum y <= 2p-2  ==>  -min_c m_c <= H(y) <= min_c' M_c'.
#   (S2) nu = uniform on {sum y in {p-1, 2p-2}} has p(p+1) points, uniform marginals and pair law
#        (1 + [a+b = p-1]) / (p(p+1)).
#   (S3) E_nu H = 0 and E_nu H^2 = V + 2/(p(p+1)) sum_{i<j} sum_a A_i(a) A_j(p-1-a).
#   (S4) the final inequality on all pairs of planes, random + structured families; min ratio vs (p-1)/(p+1).
#   (S5) exact (Fraction) equality for the ramp.
import numpy as np, itertools, sys
from fractions import Fraction
rng = np.random.default_rng(2026)

def plane_extrema(f, p):
    x1, x2 = np.meshgrid(np.arange(p), np.arange(p), indexing='ij')
    m = np.empty(p); M = np.empty(p)
    for c in range(p):
        P = f[0][x1] + f[1][x2] + f[2][(c - x1 - x2) % p]
        m[c] = -P.min(); M[c] = P.max()
    return m, M

def nu_support(p):
    pts = [y for y in itertools.product(range(p), repeat=3) if sum(y) in (p - 1, 2*p - 2)]
    return np.array(pts)

def gen(p, kind):
    fs = []
    d = int(rng.integers(1, p)); t = rng.integers(0, p, size=3)
    for i in range(3):
        if kind == 'gauss': v = rng.normal(size=p)
        elif kind == 'spike': v = np.zeros(p); v[rng.integers(p)] = rng.normal()*p; v += 0.05*rng.normal(size=p)
        elif kind == 'twoval':
            S = rng.random(p) < rng.random(); v = np.where(S, 1.0, 0.0)*rng.exponential()
        elif kind == 'ramp':        # dilated + translated sawtooth, same dilation in all slots
            v = (((np.arange(p) + t[i]) * d) % p).astype(float)
        elif kind == 'ramp_mixed':  # sawtooth with independent dilations/signs/scales
            v = (((np.arange(p) + t[i]) * int(rng.integers(1, p))) % p).astype(float) * rng.choice([-1, 1]) * rng.exponential()
        elif kind == 'line': v = rng.normal(size=p) if i == 2 else np.zeros(p)
        elif kind == 'cube': v = rng.normal(size=p)**3
        elif kind == 'ramp_noise': v = (((np.arange(p) + t[i]) * d) % p).astype(float) + 0.3*rng.normal(size=p)
        fs.append(v - v.mean())
    return fs

def run(p, trials):
    supp = nu_support(p)
    assert len(supp) == p*(p + 1), (p, len(supp))
    # (S2) marginals and pair law, exact integer counts
    for i in range(3):
        cnt = np.bincount(supp[:, i], minlength=p); assert np.all(cnt == p + 1)
    for (i, j) in [(0, 1), (0, 2), (1, 2)]:
        C = np.zeros((p, p), int)
        np.add.at(C, (supp[:, i], supp[:, j]), 1)
        a, b = np.meshgrid(np.arange(p), np.arange(p), indexing='ij')
        assert np.all(C == 1 + (a + b == p - 1)), "pair law"
    worst_thm = np.inf; worst_ratio = np.inf; worst_kind = None; worst_S1 = np.inf; worst_S3 = 0.0
    kinds = ['gauss', 'spike', 'twoval', 'ramp', 'ramp_mixed', 'line', 'cube', 'ramp_noise']
    for kind in kinds:
        for _ in range(trials):
            f = gen(p, kind)
            V_i = np.array([(g**2).mean() for g in f]); V = V_i.sum()
            if V < 1e-12: continue
            m, M = plane_extrema(f, p)
            A = [np.sort(g)[::-1] for g in f]                  # decreasing rearrangements
            H = A[0][supp[:, 0]] + A[1][supp[:, 1]] + A[2][supp[:, 2]]
            # (S1): -min_c m_c <= H <= min_c' M_c' on spt nu
            worst_S1 = min(worst_S1, (M.min() - H.max())/np.sqrt(V), (H.min() + m.min())/np.sqrt(V))
            # (S3)
            EH = H.mean(); EH2 = (H**2).mean()
            cross = sum((A[i]*A[j][::-1]).sum() for i, j in [(0, 1), (0, 2), (1, 2)])
            worst_S3 = max(worst_S3, abs(EH)/np.sqrt(V), abs(EH2 - (V + 2*cross/(p*(p + 1))))/V)
            # (S4) theorem on all pairs of planes
            lower = V - 2/(p + 1)*sum(np.sqrt(V_i[i]*V_i[j]) for i, j in [(0, 1), (0, 2), (1, 2)])
            prod = m.min()*M.min()                              # the worst pair (c, c')
            worst_thm = min(worst_thm, (prod - lower)/V)
            r = prod/V
            if r < worst_ratio: worst_ratio, worst_kind = r, kind
    print(f"p={p:3d}: (S1) min slack {worst_S1:.2e} (>=0)  (S3) max err {worst_S3:.1e}  "
          f"(S4) min (mM - bound)/V = {worst_thm:.3e} (>=0)  min mM/V = {worst_ratio:.5f} [{worst_kind}]  "
          f"(p-1)/(p+1) = {(p-1)/(p+1):.5f}", flush=True)
    return worst_thm, worst_ratio

def ramp_exact(p):
    f = [Fraction(((-x) % p)) - Fraction(p - 1, 2) for x in range(p)]
    def ext(c, sgn):
        best = None
        for x1 in range(p):
            for x2 in range(p):
                v = f[x1] + f[x2] + f[(c - x1 - x2) % p]
                v = -v if sgn < 0 else v
                best = v if best is None or v > best else best
        return best
    m1 = ext(1, -1); M2 = ext(2, +1)
    V = 3*sum(v*v for v in f)/p
    return m1, M2, V, m1*M2/V

if __name__ == '__main__':
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    for p in [3, 5, 7, 11, 13, 17, 19, 23, 29]:
        run(p, trials)
    print("(S5) ramp, exact rational arithmetic:")
    for p in [3, 5, 7, 11, 13, 17]:
        m1, M2, V, r = ramp_exact(p)
        print(f"   p={p}: m_1 = {m1}, M_2 = {M2}, V = {V}, m_1 M_2 / V = {r}  == (p-1)/(p+1) = {Fraction(p-1, p+1)} : {r == Fraction(p-1, p+1)}")
