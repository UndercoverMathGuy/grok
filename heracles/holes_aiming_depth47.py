"""Exact rational certificate for HOLES_aiming.md, Theorem DA at A=47/100,r=2/3.

No floating point, polynomial root solver, or external package in the proof.
Run: UV_CACHE_DIR=/private/tmp/heracles-uv-cache uv run --offline python heracles/holes_aiming_depth47.py
"""
from fractions import Fraction as F
from math import isqrt
from pathlib import Path
import json
import time

Q = 10**24
A = F(47, 100)
DC, WC = F(1, 100), F(3, 50)
OUT = Path(__file__).resolve().parent / "FINAL_RUNS/HOLES_aiming/depth47"


def iroot(x, n):
    lo, hi = 0, 1 << ((x.bit_length() + n - 1) // n)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if mid**n <= x:
            lo = mid
        else:
            hi = mid
    return hi if hi**n == x else lo


def sqrt_bounds(x):
    a = isqrt(x.numerator * Q**2 // x.denominator)
    lo, hi = F(a, Q), F(a + 1, Q)
    assert lo**2 <= x <= hi**2
    return lo, hi


def power_lower(x):
    """Rational lower bound for x^(2/3), independently checked by cubing."""
    assert x >= 0
    a = iroot(x.numerator**2 * Q**3 // x.denominator**2, 3)
    lo = F(a, Q)
    assert lo**3 <= x**2 < (lo + F(1, Q))**3
    return lo


def certify(box):
    d0, d1, w0, w1 = box
    if d1 <= DC and w1 <= WC:
        return "corner"
    S0 = sqrt_bounds(1 + w0*w0)[0]
    S1 = sqrt_bounds(1 + w1*w1)[1]
    target3 = 3*(1-d0)**2
    if S0**2 >= target3:
        return "trivial"
    mu = 1-d1
    b1 = max(F(0), (mu-S1*(1-A)**2)/(4*A))
    b2 = max(F(0), (mu-S1*(1-A*A))/(2*A*(1+A)))
    y = (S0-b1, b1-b2, b2)
    assert y[0] >= y[1] >= y[2] >= 0
    peel = sum(map(power_lower, y))
    if peel**3 >= target3:
        return "peel"
    D, T = d1/A, d1/A**2
    u, v = min(S1, D+2*w1), min(S1, T+2*w1)
    e2 = max(F(0), 3-2*D*u-T*v)/9
    e3 = max(F(0), 1-2*D*u-T*v-D*D*v-u*u*T)/27
    # The analytic lower bounds L2 and L3, raised to integer powers.
    if 2**9*(S0/2)**4*e2 >= target3**3:
        return "quadratic"
    if 3**81*(S0/3)**30*(e2/3)**6*e3**4 >= target3**27:
        return "cubic"
    return None


def children(box):
    d0, d1, w0, w1 = box
    dm = DC if d0 < DC < d1 else (d0+d1)/2
    wm = WC if w0 < WC < w1 else (w0+w1)/2
    return [(d0, dm, w0, wm), (dm, d1, w0, wm),
            (d0, dm, wm, w1), (dm, d1, wm, w1)]


def main():
    start = time.time()
    # Exact analytic-corner checks. These brackets decrease on the corner.
    a1, a2 = 2/A+1/A**2, 2/A**2+1/A**4
    B = 6-DC*(3+a2)-2*WC*a1
    # Drop the positive d^2 term, so all remaining terms decrease.
    C = 3-3*DC-DC*a2-2*WC*a1-(DC/A**2)*(DC/A**2+2*WC)-(DC/A+2*WC)**2/A**2
    assert B > 0 and C > 0
    root = (F(0), F(1, 2), F(0), F(3, 2))
    stack = [("", root)]
    leaves = {}
    visited = 0
    while stack:
        path, box = stack.pop()
        visited += 1
        assert visited <= 200000 and len(path) <= 24, (path, box)
        reason = certify(box)
        if reason:
            leaves[path] = {"box": list(map(str, box)), "reason": reason}
        else:
            stack.extend((path+str(i), child) for i, child in enumerate(children(box)))
    # Replay the saved partition from its root: no missing or overlapping interior cells.
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "certificate.json"
    dest.write_text(json.dumps(leaves, indent=2)+"\n")
    saved = json.loads(dest.read_text())
    seen = set()
    def replay(path, box):
        if path in saved:
            record = saved[path]
            assert list(map(str, box)) == record["box"]
            assert certify(box) == record["reason"]
            seen.add(path)
            return
        assert len(path) <= 24
        for i, child in enumerate(children(box)):
            replay(path+str(i), child)
    replay("", root)
    assert seen == set(saved)
    counts = {r: sum(v["reason"] == r for v in saved.values())
              for r in sorted({v["reason"] for v in saved.values()})}
    result = {"status": "PASS", "arithmetic": "exact integers and fractions",
              "A": "47/100", "r": "2/3", "rectangle": list(map(str, root)),
              "corner_brackets": [str(B), str(C)], "leaves": len(saved),
              "visited": visited, "replayed": len(seen), "counts": counts,
              "root_denominator": Q, "seconds": time.time()-start}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
