"""robust_cert.py -- rigorous kappa-robustness certificates for the symmetric-class argmax (Theorem R).

Setting: programme in y-form, f_v(kappa) = sum_pi c_pi(kappa) v_pi^r over the (exact, rational) vertices v of
P = { y >= 0 : cov^T y >= 1 }, with c_pi = kappa_{d_pi} d_pi^{q/2} for real-type irreps (kappa_1 = 1 for real characters).
f_v is LINEAR in the unknown vector kappa = (kappa_d)_d. The minimiser over vertices is the programme's optimum (Lemma V).

Claim tested on a box B = prod_d [lo_d, hi_d] of admissible kappa (Theorems K1-K3):
    (COUNT)    every minimising vertex has support size K0          ("the count is kappa-free on B")
    (IDENTITY) the minimising vertex is v0                           (only where it holds)
Certificate: recursive bisection of B; a sub-box is closed when for every 'bad' vertex w there is a 'good' vertex v with
f_v <= f_w at all corners of the sub-box (linear => on the whole sub-box). Arithmetic in mpmath with 40 digits; every
closing inequality must hold with slack > 1e-25 (reported). Output: out/robust_cert.log / .json
"""
import itertools, json, sys
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gt_core as gc  # noqa: E402

mp.mp.dps = 40
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)


def setup(gname, nu, q):
    G = gc.get(gname); irr, _ = gc.programme_data(G)
    cov = np.array([[Fraction(x).limit_denominator(10 ** 6) for x in p["cov"]] for p in irr], dtype=object)
    exact = all(abs(float(Fraction(x)) - y) < 1e-12 for row, p in zip(cov, irr) for x, y in zip(row, p["cov"]))
    if not exact:
        raise SystemExit(f"{gname}: non-rational characters; certificate script needs exact vertices")
    verts = gc.vertices(cov, exact=True)
    r = mp.mpf(q) / nu
    ds = sorted(set(p["d"] for p in irr if p["d"] > 1))
    # f_v(kappa) = a_v + sum_d b_{v,d} kappa_d  (a_v from real characters)
    lin = []
    for v in verts:
        a = mp.mpf(0); b = {d: mp.mpf(0) for d in ds}
        for p, x in zip(irr, v):
            if x == 0:
                continue
            term = mp.mpf(p["d"]) ** (mp.mpf(q) / 2) * (mp.mpf(x.numerator) / x.denominator) ** r
            if p["d"] == 1:
                a += term
            else:
                b[p["d"]] += term
        lin.append((a, b))
    return G, irr, verts, ds, lin, r


def corners(box):
    ds = sorted(box)
    for bits in itertools.product([0, 1], repeat=len(ds)):
        yield {d: box[d][bit] for d, bit in zip(ds, bits)}


def fval(l, kap):
    a, b = l
    return a + sum(b[d] * kap[d] for d in b)


def certify(lin, good, bad, box, depth=0, maxdepth=14, stats=None):
    """True if on every point of box some good vertex beats every bad vertex (per bad vertex, one dominating good one per sub-box)."""
    stats = stats if stats is not None else dict(boxes=0, min_slack=mp.inf)
    cs = list(corners(box))
    ok = True
    for w in bad:
        found = False
        for v in good:
            sl = min(fval(lin[w], k) - fval(lin[v], k) for k in cs)
            if sl > mp.mpf("1e-25"):
                found = True; stats["min_slack"] = min(stats["min_slack"], sl); break
        if not found:
            ok = False; break
    if ok:
        stats["boxes"] += 1
        return True, stats
    if depth >= maxdepth:
        return False, stats
    # bisect the widest side (log scale)
    d = max(box, key=lambda d: mp.log(box[d][1] / box[d][0]))
    lo, hi = box[d]; mid = mp.sqrt(lo * hi)
    b1 = dict(box); b1[d] = (lo, mid); b2 = dict(box); b2[d] = (mid, hi)
    r1, stats = certify(lin, good, bad, b1, depth + 1, maxdepth, stats)
    if not r1:
        return False, stats
    return certify(lin, good, bad, b2, depth + 1, maxdepth, stats)


def run(gname, nu, q, count=None, ident=None, box_mode="odd"):
    G, irr, verts, ds, lin, r = setup(gname, nu, q)
    if box_mode == "odd":
        box = {d: (mp.mpf(d) ** ((nu - 1) * (1 - r)), mp.mpf(d) ** (nu * (1 - r))) for d in ds}
    else:
        box = box_mode(ds, r)
    sizes = [sum(1 for x in v if x != 0) for v in verts]
    names = [tuple(gc.support_names(irr, v)) for v in verts]
    res = {}
    if count is not None:
        good = [i for i in range(len(verts)) if sizes[i] == count]
        bad = [i for i in range(len(verts)) if sizes[i] != count]
        ok, st = certify(lin, good, bad, box)
        res["count"] = dict(K=count, certified=ok, boxes=st["boxes"], min_slack=float(st["min_slack"]) if ok else None)
        log(f"  {gname} nu {nu} q {q:g}: COUNT K = {count} on box {{{', '.join(f'{d}: [{float(box[d][0]):.4f}, {float(box[d][1]):.4f}]' for d in ds)}}}: "
            f"{'CERTIFIED' if ok else 'NOT certified'} ({st['boxes']} sub-boxes, min slack {float(st['min_slack']):.3e})")
    if ident is not None:
        good = [i for i in range(len(verts)) if names[i] == tuple(ident)]
        bad = [i for i in range(len(verts)) if names[i] != tuple(ident)]
        ok, st = certify(lin, good, bad, box)
        res["identity"] = dict(support=ident, certified=ok)
        log(f"  {gname} nu {nu} q {q:g}: IDENTITY {ident} on the same box: {'CERTIFIED' if ok else 'NOT certified (fails somewhere in the box)'}")
    return res, irr, verts, lin, ds, r


def identity_region(gname, nu, q, ident, kfix_d):
    """exact condition for ident to be optimal as linear inequalities in kappa (reported for the single-irrep rivals)."""
    G, irr, verts, ds, lin, r = setup(gname, nu, q)
    names = [tuple(gc.support_names(irr, v)) for v in verts]
    v0 = names.index(tuple(ident))
    out = []
    for i, nm in enumerate(names):
        if i == v0 or len(nm) != 1:
            continue
        # f_i - f_v0 >= 0 with both single irreps: b_i kappa_{d_i} >= b_0 kappa_{d_0}
        (a0, b0), (a1, b1) = lin[v0], lin[i]
        d0 = [d for d in ds if b0[d] != 0][0]; d1 = [d for d in ds if b1[d] != 0]
        if not d1:
            continue
        d1 = d1[0]
        ratio = b0[d0] / b1[d1]
        out.append((nm[0], d1, d0, float(ratio)))
        log(f"  {gname} nu {nu} q {q:g}: {ident[0]} beats {nm[0]} iff kappa_{d1} / kappa_{d0} >= {float(ratio):.6f}")
    return out


def main():
    out = {}
    log("== S5")
    for nu, q in [(3, 2.0), (5, 2.0), (3, 2.5)]:
        res, *_ = run("s5", nu, q, count=1, ident=("std_sign",))
        out[f"s5_nu{nu}_q{q:g}"] = res
        identity_region("s5", nu, q, ("std_sign",), None)
    log("   provable relations (restriction lemma, Theorem K5): kappa_D / kappa_k >= (k/D)^{q-1}; at q = 2: kappa_5/kappa_4 >= 0.8, kappa_6/kappa_4 >= 0.6667")
    log("== S4")
    for nu, q in [(3, 2.0), (3, 2.5), (5, 2.0)]:
        for K in (2, 3):
            res, *_ = run("s4", nu, q, count=K)
            out[f"s4_nu{nu}_q{q:g}_K{K}"] = res
    (HERE / "out" / "robust_cert.json").write_text(json.dumps(out, indent=1, default=str))
    (HERE / "out" / "robust_cert.log").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
