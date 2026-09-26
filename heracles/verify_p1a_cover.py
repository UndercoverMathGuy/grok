r"""VERIFY / P1A -- independent coverage audit of p1a_cert.py's branch-and-bound.

Does NOT modify p1a_cert.py.  It re-runs the identical box-splitting loop (importing
p1a_cert.G_lower, so the *bound* is theirs) while RECORDING every box and its fate:

    'cert'    the box was closed by the bound  (G_lower.a >= tgt.b)
    'skip'    the box was DISCARDED as (KL)-infeasible because (bh+th).b < D_n(n).a
    'split'   the box was bisected

Then an independent checker verifies that

  (i)  the union of {cert boxes} u {skip boxes} covers the whole starting rectangle
       (leaf boxes tile it; overlaps allowed, gaps are not), and
  (ii) reports the AREA fraction carried by the 'skip' rule, i.e. how much of the
       certificate is really being done by Theorem E at h = n rather than by the bound.

Also re-runs the BB with the skip rule DISABLED, to see whether the certificate
survives without that inherited ingredient.

Run:  uv run --with mpmath python verify_p1a_cover.py
"""
from __future__ import annotations

import json
import os

from mpmath import iv, mp

import p1a_cert as PC

iv.dps = 30
mp.dps = 30

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "VERIFY")


def bb_record(n, r, b_hi, t_lo, t_hi, tgt, Ds, max_boxes=2_000_000, staircase=False,
              use_skip=True):
    """p1a_cert.certify_rectangle, verbatim logic, but recording every leaf box."""
    stack = [(iv.mpf(0), b_hi, t_lo, t_hi)]
    leaves = []          # (bl, bh, tl, th, fate) as floats
    done = 0
    while stack:
        bl, bh, tl, th = stack.pop()
        done += 1
        if done > max_boxes:
            return False, done, leaves
        box = (float(bl.a), float(bh.b), float(tl.a), float(th.b))
        if use_skip and (bh + th).b < Ds[n].a:
            leaves.append(box + ("skip",))
            continue
        g = PC.G_lower(n, r, bl, bh, tl, th, Ds, tgt, staircase=staircase)
        if g.a >= tgt.b:
            leaves.append(box + ("cert",))
            continue
        wb = float(bh.b) - float(bl.a)
        wt = float(th.b) - float(tl.a)
        if wb < 1e-13 and wt < 1e-13:
            leaves.append(box + ("FAIL",))
            return False, done, leaves
        if wb >= wt:
            mid = (bl + bh) / 2
            stack.append((bl, mid, tl, th))
            stack.append((mid, bh, tl, th))
        else:
            mid = (tl + th) / 2
            stack.append((bl, bh, tl, mid))
            stack.append((bl, bh, mid, th))
    return True, done, leaves


def coverage_check(leaves, B, T0, T1, ngrid=1200):
    """Independent: sample a fine grid of the rectangle [0,B] x [T0,T1] and check every
    sample lies in at least one leaf box.  Also exact area accounting."""
    bad = []
    import numpy as np
    bs = np.linspace(0.0, B, ngrid)
    ts = np.linspace(T0, T1, ngrid)
    arr = np.array([[l[0], l[1], l[2], l[3]] for l in leaves])
    BB, TT = np.meshgrid(bs, ts, indexing="ij")
    covered = np.zeros(BB.shape, dtype=bool)
    for (bl, bh, tl, th) in arr:
        covered |= (BB >= bl - 1e-15) & (BB <= bh + 1e-15) & (TT >= tl - 1e-15) & (TT <= th + 1e-15)
    if not covered.all():
        idx = np.argwhere(~covered)[:5]
        bad = [(float(bs[i]), float(ts[j])) for i, j in idx]
    area = {}
    for l in leaves:
        area[l[4]] = area.get(l[4], 0.0) + (l[1] - l[0]) * (l[3] - l[2])
    return covered.all(), bad, area, (B * (T1 - T0))


def main():
    os.makedirs(OUT, exist_ok=True)
    rec = {}
    for n, q in ((2, 2.0), (3, 2.0)):
        r = q / (n + 1.0)
        tgt = PC.ipow(iv.mpf(n + 1), 1.0 - r)
        Ds = [PC.D_h(n, h) for h in range(n + 1)]
        b_hi = iv.mpf(1) / (n + 1)
        e = PC.eps_cert(n, r, tgt)
        C, _ = PC.C_const(n, r)
        t_hi = iv.mpf(float(PC.ipow(tgt / C, 1.0 / r).b))
        key = f"n{n}_q{q}"
        print(f"\n=== n={n} q={q} r={r:.12f} ===")
        for use_skip in (True, False):
            ok, done, leaves = bb_record(n, r, b_hi, iv.mpf(e), t_hi, tgt, Ds,
                                         max_boxes=400_000, use_skip=use_skip)
            lab = "with D_n skip" if use_skip else "WITHOUT D_n skip"
            if not ok:
                print(f"  {lab}: NOT CERTIFIED (boxes tried {done})")
                rec[key + ("_skip" if use_skip else "_noskip")] = dict(certified=False, boxes=done)
                continue
            full, bad, area, tot = coverage_check(leaves, float(b_hi.a), e, float(t_hi.b))
            print(f"  {lab}: certified, {len(leaves)} leaves; coverage complete = {full}"
                  + ("" if full else f"  UNCOVERED SAMPLES {bad}"))
            print(f"    rectangle area {tot:.6f};  " +
                  ", ".join(f"{k} {v:.6f} ({100*v/tot:.1f}%)" for k, v in sorted(area.items())))
            rec[key + ("_skip" if use_skip else "_noskip")] = dict(
                certified=True, leaves=len(leaves), coverage_complete=bool(full),
                area=area, rect_area=tot, eps_cert=e, tau_hi=float(t_hi.b))
    with open(os.path.join(OUT, "p1a_cover.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"\nwrote {os.path.join(OUT,'p1a_cover.json')}")


if __name__ == "__main__":
    main()
