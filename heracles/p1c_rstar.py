"""P1C / step 4e: the RIGOROUS r-map.

ADVERSARY.md Sec 6 reaches the same two-variable programme as Theorem H here and locates its
crossover `r*(n)` by minimising on a 600 x 600 grid plus a Nelder-Mead polish, explicitly
flagging that "a certified version would be a routine interval branch-and-bound in two
variables, which I did not run".  This script runs the certified version: it bisects on `r`
using the CELL-WISE guaranteed lower bound of p1c_certify.py, so the reported `r*` is a value
below which the programme is *proved* to clear the target, not merely measured to.

Because the cell bound is conservative, a coarser grid gives a weaker but still VALID bound:
if it closes, it closes.  The reported `r*` is therefore a rigorous lower bound on the true
crossover of the programme.
"""
import argparse, json, os
import numpy as np
from p1c_certify import certify

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "P1C")


def closes(n, r, nR, nS):
    q = r * (n + 1)
    rec = certify(n, q, nR, nS)
    return rec["cert_ratio"] >= 1.0, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", nargs="+", type=int, default=[2, 3, 4, 5])
    ap.add_argument("--nR", type=int, default=160)
    ap.add_argument("--nS", type=int, default=160)
    ap.add_argument("--iters", type=int, default=9)
    ap.add_argument("--tag", default="rstar")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for n in a.ns:
        lo = 2.0 / (n + 1)                      # q = 2, plain weight decay
        ok_lo, rec_lo = closes(n, lo, a.nR, a.nS)
        print(f"n={n}: q=2 (r={lo:.5f}) -> ratio {rec_lo['cert_ratio']:.6f} "
              f"{'CLOSES' if ok_lo else 'does NOT close'}", flush=True)
        if not ok_lo:
            rows.append(dict(n=n, r_q2=lo, ratio_q2=rec_lo["cert_ratio"], rstar=None))
            continue
        hi = 1.0
        trace = [(lo, rec_lo["cert_ratio"], True)]
        for _ in range(a.iters):
            m = 0.5 * (lo + hi)
            ok, rec = closes(n, m, a.nR, a.nS)
            trace.append((m, rec["cert_ratio"], bool(ok)))
            print(f"    r={m:.5f} (q={m*(n+1):.4f}) -> ratio {rec['cert_ratio']:.6f} "
                  f"{'ok' if ok else 'short'}", flush=True)
            if ok:
                lo = m
            else:
                hi = m
        rows.append(dict(n=n, r_q2=2.0 / (n + 1), ratio_q2=rec_lo["cert_ratio"],
                         rstar_certified=lo, rstar_bracket=[lo, hi],
                         qstar_certified=lo * (n + 1), grid=[a.nR, a.nS],
                         trace=[[float(x), float(y), bool(z)] for x, y, z in trace]))
        print(f"n={n}: CERTIFIED r* >= {lo:.5f}  (q* >= {lo*(n+1):.4f}); "
              f"bracket [{lo:.5f}, {hi:.5f}]", flush=True)
    with open(os.path.join(OUT, f"{a.tag}.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
