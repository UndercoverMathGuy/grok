"""lemma_family3.py -- the exact 2-parameter family of 3-atom junk-free configurations at n = 2.

Setting (see LEMMA_ATTEMPT.md Sec. 2/3).  n = 2 operands, nu = 3.  A "pure" neuron l at one frequency is
described by a complex clean amplitude c_l and two unit-modulus phase markers x_l, y_l (one per operand).
Junk-freeness of a population {(c_l, x_l, y_l)} with clean amplitude normalised to 1 is the linear system

    sum_l c_l = 1,      sum_l c_l x_l = 0,      sum_l c_l y_l = 0,      sum_l c_l x_l y_l = 0.

Plain language: the four sums are the four Fourier "sign patterns" the population emits at its frequency;
the first is the useful one, the other three are junk and must cancel.

With N = 3 atoms and the gauge x_1 = y_1 = 1 the system has a TWO-REAL-PARAMETER family of solutions:
for phases (x_2, x_3, y_2, y_3) subject to ONE complex equation

    R (x_2 - 1)/(y_2 - 1) + S (x_3 - 1)/(y_3 - 1) = -1,   R = x_3/(x_2 - x_3),  S = -x_2/(x_2 - x_3),

the amplitudes are forced:  c_2 = R/(y_2 - 1),  c_3 = S/(y_3 - 1),  c_1 = 1 - c_2 - c_3.
The (n+1)-block is the point x_l = y_l = omega^{l-1}, omega = e^{2 pi i/3}, where c_l = 1/3 for all l.

This script walks that family (solving the complex equation for (tau_2, tau_3) at each (sigma_2, sigma_3))
and reports  F_r = sum_l |c_l|^r  against the conjectured minimum 3^{1-r}.  If F_r ever dips below,
the (n+1)-block is NOT optimal and the target inequality is refuted.

Usage: uv run python heracles/lemma_family3.py
"""
import json, math
from pathlib import Path

import numpy as np
from scipy.optimize import fsolve

OUT = Path(__file__).resolve().parent / "FINAL_RUNS" / "LEMMA"


def amps(s2, s3, t2, t3):
    x2, x3, y2, y3 = (np.exp(1j * u) for u in (s2, s3, t2, t3))
    den = x2 - x3
    if abs(den) < 1e-12 or abs(y2 - 1) < 1e-12 or abs(y3 - 1) < 1e-12:
        return None
    R = x3 / den
    S = -x2 / den
    c2 = R / (y2 - 1)
    c3 = S / (y3 - 1)
    c1 = 1 - c2 - c3
    resid = R * (x2 - 1) / (y2 - 1) + S * (x3 - 1) / (y3 - 1) + 1.0
    return np.array([c1, c2, c3]), resid


def resid_tau(t, s2, s3):
    out = amps(s2, s3, t[0], t[1])
    if out is None:
        return np.array([1e3, 1e3])
    return np.array([out[1].real, out[1].imag])


def verify(c, s2, s3, t2, t3):
    """Independent check of all four sums."""
    x = np.array([1.0, np.exp(1j * s2), np.exp(1j * s3)])
    y = np.array([1.0, np.exp(1j * t2), np.exp(1j * t3)])
    return np.array([abs((c).sum() - 1), abs((c * x).sum()), abs((c * y).sum()), abs((c * x * y).sum())]).max()


if __name__ == "__main__":
    w = 2 * math.pi / 3
    rows = []
    for r in (2 / 3, 5 / 6, 1.0, 0.5):
        target = 3 ** (1 - r)
        best = (np.inf, None)
        nsol = 0
        # dense scan of (sigma_2, sigma_3) around, and far from, the symmetric point
        for s2 in np.linspace(0.05, 2 * math.pi - 0.05, 121):
            for s3 in np.linspace(0.05, 2 * math.pi - 0.05, 121):
                if abs(s2 - s3) < 1e-3:
                    continue
                for t0 in ((w, 2 * w), (2 * w, w), (s2, s3), (1.0, 4.0)):
                    t, info, ier, msg = fsolve(resid_tau, np.array(t0), args=(s2, s3), full_output=True)
                    if ier != 1:
                        continue
                    out = amps(s2, s3, t[0], t[1])
                    if out is None or abs(out[1]) > 1e-10:
                        continue
                    c = out[0]
                    if verify(c, s2, s3, t[0], t[1]) > 1e-9:
                        continue
                    nsol += 1
                    F = float((np.abs(c) ** r).sum())
                    if F < best[0]:
                        best = (F, (s2, s3, float(t[0]), float(t[1]), np.abs(c).tolist()))
        print(f"r = {r:.4f}   solutions found {nsol}   min F_r = {best[0]:.10f}   3^(1-r) = {target:.10f}"
              f"   ratio {best[0]/target:.8f}")
        print(f"          argmin: sigma {best[1][0]:.4f} {best[1][1]:.4f}  tau {best[1][2]:.4f} {best[1][3]:.4f}"
              f"  |c| = {[round(u,6) for u in best[1][4]]}")
        rows.append(dict(r=r, minF=best[0], target=target, ratio=best[0] / target, arg=best[1], nsol=nsol))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "family3.json").write_text(json.dumps(rows, indent=1))
