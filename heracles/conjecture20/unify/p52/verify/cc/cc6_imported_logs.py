"""Referee: (a) adversarial numerical check of the imported payments used by F (Note Thm c1sharp, Part I Thm A),
(b) check of the payment values F printed in the official certificate logs against true minimal payments.

(a) minimise P_1(s) + c_1(s) max(|B|,|C|,|D|) and P_1(s) + sqrt(|B|^2+|C|^2+|D|^2)/(2 sqrt 3) over atom systems with
    A = 1 (SLSQP, epigraph form, K = 1..6 atoms, many starts); both minima must be >= 1.
(b) for random Region-II boxes of cert_l*.log, random points inside: true min P_1 (SLSQP) >= logged F - 1e-4.
"""
import sys, os, re, time, warnings
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy.optimize import minimize
from cc2_payment import min_P1

rng = np.random.default_rng(11)


def imported_min(s, kind, K, starts=20):
    r = 1 / s; c1 = 1 - 3 ** (1 - s)
    best = np.inf

    def unpack(z):
        rho = z[:K] ** 2; a = z[K:2 * K]; b = z[2 * K:3 * K]; c = z[3 * K:4 * K]
        be = rho * np.exp(1j * a); xi = np.exp(1j * b); eta = np.exp(1j * c)
        return be, xi, eta

    def owns(z):
        be, xi, eta = unpack(z)
        return be.sum(), (be * xi).sum(), (be * eta).sum(), (be * xi * eta).sum()

    def P1(z):
        return 3 ** (1 - s) * np.sum((z[:K] ** 2 + 1e-14) ** r) ** s

    for st in range(starts):
        z0 = np.concatenate([rng.uniform(0.2, 1, K), rng.uniform(-np.pi, np.pi, 3 * K), [rng.uniform(0, 1)]])
        if kind == 'c1':
            obj = lambda z: P1(z[:-1]) + c1 * z[-1]
            cons = [{'type': 'eq', 'fun': lambda z: np.array([owns(z[:-1])[0].real - 1, owns(z[:-1])[0].imag])},
                    {'type': 'ineq', 'fun': lambda z: np.array([z[-1] - abs(v) for v in owns(z[:-1])[1:]])}]
        else:
            obj = lambda z: P1(z[:-1]) + np.sqrt(sum(abs(v) ** 2 for v in owns(z[:-1])[1:])) / (2 * np.sqrt(3))
            cons = [{'type': 'eq', 'fun': lambda z: np.array([owns(z[:-1])[0].real - 1, owns(z[:-1])[0].imag])}]
        res = minimize(obj, z0, method='SLSQP', constraints=cons, options=dict(maxiter=600, ftol=1e-13))
        A = owns(res.x[:-1])[0]
        if abs(A - 1) < 1e-8:
            if kind == 'c1':
                val = P1(res.x[:-1]) + c1 * max(abs(v) for v in owns(res.x[:-1])[1:])
            else:
                val = res.fun
            best = min(best, val)
    return best


if __name__ == '__main__':
    t0 = time.time()
    worst = {}
    for kind in ('c1', 'thmA'):
        for s in (1.0, 1.1, 1.25, 1.5):
            v = min(imported_min(s, kind, K) for K in (1, 2, 3, 4, 6))
            worst[(kind, s)] = v
            print(f"{kind} s={s}: min over atoms of [P_1 + payment-term] with A=1 = {v:.6f}  (must be >= 1)", flush=True)
    # (b) logged boxes
    P52 = os.path.abspath(os.path.join(HERE, '..', '..'))
    boxes = []
    for fn in ('cert_l0_7.log', 'cert_l1_6.log', 'cert_l2_5.log', 'cert_l3_4.log'):
        for line in open(os.path.join(P52, fn)):
            if line.startswith('II '):
                mm = re.match(r"II \[(.*?)\] \[(.*?)\] \[(.*?),(.*?)\] F=([\d.]+) worst=([\d.]+)", line)
                lo = np.array([float(v) for v in mm.group(1).split(',')]); hi = np.array([float(v) for v in mm.group(2).split(',')])
                boxes.append((lo, hi, float(mm.group(3)), float(mm.group(4)), float(mm.group(5)), float(mm.group(6))))
    print(f"parsed {len(boxes)} Region-II boxes; F range [{min(b[4] for b in boxes):.4f}, {max(b[4] for b in boxes):.4f}]")
    idx = rng.choice(len(boxes), 30, replace=False)
    nviol = 0; mingap = np.inf
    for i in idx:
        lo, hi, pl, ph, F, w = boxes[i]
        for _ in range(2):
            m = lo + rng.uniform(size=3) * (hi - lo)
            ang = rng.uniform(pl, ph)
            Pm = min_P1(m[0], m[1], m[2], ang, K=5, starts=10)
            gap = Pm - F; mingap = min(mingap, gap)
            if gap < -1e-4:
                nviol += 1; print("VIOLATION", lo, hi, pl, ph, F, m, ang, Pm)
    print(f"logged-F check: {len(idx)*2} points, violations {nviol}, min(Pmin - F_logged) = {mingap:.5f}; time {time.time()-t0:.0f}s")
