"""Branch-and-bound certificate for the (5,2) single-feature bound at s = 3/2 (b = 2).

Claim certified:  for every own-data point (|A| = 1, A = e^{i alpha}, B = m1 >= 0, C = m2 >= 0, D = m12 e^{i phi}),
   min( trivial bound, HS-certificate bound ) / (Gamma sqrt(F^2 + R^2/12))  <=  TARGET < 1   for all R >= 0,
where F is the rigorous payment lower bound (pay_adapt.payment).  Reductions: alpha >= 0 (conjugation), m1 >= m2
(slot swap), torus normalisation (B, C real >= 0) via equivariance of the HS family; phi_pay = phi + alpha.
Outer boxes: (m1, m2, m12, phi_pay); inner: alpha-cells.
Usage: python bb52.py TARGET logfile [region]
"""
import sys, time, json, heapq
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/p52')
import numpy as np
from iv import I, dn, up, cos_iv, COS72, SQ2
from ref3_pay_adaptp import payment
from cert_hs import cert_data, eval_cells, GAMMA
from hs2 import elim_ratio

TARGET = float(sys.argv[1]) if len(sys.argv) > 1 else 0.995
LOG = sys.argv[2] if len(sys.argv) > 2 else 'bb52.log'
MMAX = 1.40
KA = I(1.0) - COS72                                  # kappa_min = 1 - cos(2 pi/5)
THR_I = (KA / (GAMMA * TARGET)).hi                   # F >= THR_I  =>  trivial bound certifies
MINW = 2e-3

logf = open(LOG, 'a')


def log(msg):
    logf.write(msg + "\n"); logf.flush()


def alpha_max(F):
    """smallest alpha_m in [0, pi/2] such that KA cos(alpha)/(Gamma F) <= TARGET for alpha >= alpha_m (rigorous)."""
    # need cos(alpha) <= TARGET Gamma F / KA ; take c = lower bound of RHS
    c = (I(TARGET) * GAMMA * I(F) / KA).lo
    if c >= 1.0:
        return 0.0
    a = float(np.arccos(max(min(c, 1.0), -1.0))) + 1e-9
    # verify: cos(a) <= c rigorously (cos decreasing on [0, pi])
    while cos_iv(np.array([a]), np.array([a])).hi[0] > c:
        a += 1e-6
    return min(a, np.pi / 2)


def try_certs(certs, a_lo, a_hi, m_lo, m_hi, p_lo, p_hi, F):
    ph_lo = dn(p_lo - a_hi); ph_hi = up(p_hi - a_lo)   # REF3: outward rounding
    best = np.inf
    for cd in certs:
        r = eval_cells(cd, np.array([a_lo]), np.array([a_hi]), m_lo[None, :], m_hi[None, :],
                       np.array([ph_lo]), np.array([ph_hi]), np.array([F]))[0]
        best = min(best, r)
        if best <= TARGET:
            return best, cd
    return best, None


def new_cert(a_c, m_c, p_c, F):
    val, Gs = elim_ratio(a_c, m_c[0], m_c[1], m_c[2], p_c - a_c, F, deg=2, ret=True)
    return cert_data(Gs), val


POOL = []          # global pool of recent certificates (tried before solving a new SDP)


def pool_add(cd):
    POOL.append(cd)
    if len(POOL) > 40:
        POOL.pop(0)


def region2(m_lo, m_hi, p_lo, p_hi, F, certs, depth_max=4):
    """certify all alpha in [0, alpha_max(F)] with HS certificates; returns (ok, certs_used, worst)."""
    am = alpha_max(F)
    stack = [(0.0, am, 0)]
    used = []; worst = 0.0
    m_c = 0.5 * (m_lo + m_hi); p_c = 0.5 * (p_lo + p_hi)
    while stack:
        a_lo, a_hi, d = stack.pop()
        # trivial bound with cos(alpha_lo)
        triv = (KA * cos_iv(np.array([a_lo]), np.array([a_hi])).clip0() / (GAMMA * I(F))).hi[0]
        if triv <= TARGET:
            worst = max(worst, triv); continue
        r, cd = try_certs(used + certs + POOL[::-1], a_lo, a_hi, m_lo, m_hi, p_lo, p_hi, F)
        if cd is None:
            try:
                cdn, val = new_cert(0.5 * (a_lo + a_hi), m_c, p_c, F)
            except Exception as ex:
                return False, used, np.inf
            pool_add(cdn)
            r2, _ = try_certs([cdn], a_lo, a_hi, m_lo, m_hi, p_lo, p_hi, F)
            if r2 <= TARGET:
                used.append(cdn); worst = max(worst, r2); continue
            r = min(r, r2)
            if d < depth_max and val < TARGET:
                am_ = 0.5 * (a_lo + a_hi)
                stack.append((a_lo, am_, d + 1)); stack.append((am_, a_hi, d + 1))
                used.append(cdn)
                continue
            return False, used, r
        worst = max(worst, r)
    return True, used, worst


def main():
    t0 = time.time()
    # initial boxes
    mg = np.linspace(0, MMAX, 8); pg = np.linspace(-np.pi, np.pi, 9); pg[0] = -3.1415926535897936; pg[-1] = 3.1415926535897936   # REF3: cover true [-pi, pi]
    boxes = []
    sel = None; lsel = None
    if len(sys.argv) > 3:
        if sys.argv[3].startswith('l='):
            lsel = [int(v) for v in sys.argv[3][2:].split(',')]      # restrict phi_pay cells
        else:
            sel = [int(v) for v in sys.argv[3].split(',')]     # restrict to initial cells i,j,k,l
    for i in range(7):
        for j in range(7):
            for k in range(7):
                if sel is not None and (i, j, k) != tuple(sel[:3]):
                    continue
                m_lo = np.array([mg[i], mg[j], mg[k]]); m_hi = np.array([mg[i + 1], mg[j + 1], mg[k + 1]])
                if m_hi[0] < m_lo[1]:
                    continue                      # m1 >= m2 symmetry
                for l in range(8):
                    if sel is not None and l != sel[3]:
                        continue
                    if lsel is not None and l not in lsel:
                        continue
                    boxes.append((m_lo, m_hi, pg[l], pg[l + 1], []))
    nI = nII = nfail = 0; ncert = 0; worst2 = 0.0
    log(f"start TARGET={TARGET} THR_I={THR_I:.6f} boxes={len(boxes)}")
    while boxes:
        m_lo, m_hi, p_lo, p_hi, certs = boxes.pop()
        F = payment(m_lo, m_hi, p_lo, p_hi, need=THR_I)
        if F >= THR_I:
            nI += 1
            continue
        ok, used, worst = region2(m_lo, m_hi, p_lo, p_hi, F, certs)
        ncert += len(used)
        if ok:
            nII += 1; worst2 = max(worst2, worst)
            log(f"II {np.round(m_lo,4).tolist()} {np.round(m_hi,4).tolist()} [{p_lo:.4f},{p_hi:.4f}] F={F:.4f} worst={worst:.4f}")
            continue
        # split
        wm = m_hi - m_lo; wp = (p_hi - p_lo) / 3.0
        w = np.concatenate([wm, [wp]])
        if w.max() < MINW:
            nfail += 1
            log(f"FAIL {m_lo.tolist()} {m_hi.tolist()} [{p_lo},{p_hi}] F={F:.5f} r={worst:.5f}")
            continue
        k = int(np.argmax(w))
        if k < 3:
            mid = 0.5 * (m_lo[k] + m_hi[k])
            a_hi = m_hi.copy(); a_hi[k] = mid
            b_lo = m_lo.copy(); b_lo[k] = mid
            ch = [(m_lo, a_hi, p_lo, p_hi), (b_lo, m_hi, p_lo, p_hi)]
        else:
            pm = 0.5 * (p_lo + p_hi)
            ch = [(m_lo, m_hi, p_lo, pm), (m_lo, m_hi, pm, p_hi)]
        for c in ch:
            if c[1][0] < c[0][1]:
                continue
            boxes.append((c[0], c[1], c[2], c[3], (certs + used)[-6:]))
        if (nI + nII) % 200 == 0:
            log(f"progress: I={nI} II={nII} fail={nfail} queue={len(boxes)} certs={ncert} worstII={worst2:.4f} t={time.time()-t0:.0f}s")
    log(f"DONE: I={nI} II={nII} fail={nfail} certs={ncert} worstII={worst2:.4f} time={time.time()-t0:.0f}s")
    print(f"DONE: I={nI} II={nII} fail={nfail} certs={ncert} worstII={worst2:.4f} time={time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
