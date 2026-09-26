"""Independent numerical checks of the logical reductions in ARGUMENT.md.

These are consistency checks. Exact ansatz certification is in vertices.py.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from raw_margin import setup

ROOT = Path(__file__).resolve().parent


def vertex_amplitudes(p):
    record = json.loads((ROOT / f"vertices_p{p}.json").read_text())
    alpha = 2 * np.cos(2 * np.pi / p)
    from fractions import Fraction
    x = np.array([np.polyval([float(Fraction(c)) for c in coeffs], alpha)
                  for coeffs in record["amplitudes_descending_polynomial_coefficients"]])
    x[np.abs(x) < 1e-12] = 0
    return x


def ansatz(p, x):
    nu = 3
    support = np.flatnonzero(x)
    cost = np.sum(x[support] ** (2 / 3))
    shares = x[support] ** (2 / 3) / cost
    weights = np.zeros((nu, nu * len(support), p))
    for k, b, offset in zip(support + 1, shares, range(0, weights.shape[1], nu)):
        for ell in range(nu):
            phi = np.pi * ell / nu
            for j in range(2):
                weights[j, offset + ell] = np.sqrt(b) * np.cos(2*np.pi*k*np.arange(p)/p + phi)
            weights[2, offset + ell] = np.sqrt(b) * np.cos(2*np.pi*k*np.arange(p)/p + 2*phi)
    return weights, cost ** (-1.5)


def check(p, starts=4):
    x = vertex_amplitudes(p)
    a = 1 - np.cos(2*np.pi*np.outer(np.arange(1, (p+1)//2), np.arange(1, (p+1)//2))/p)
    assert np.min(a @ x) >= 1 - 1e-11
    weights, gamma = ansatz(p, x)
    ev, con, jac, norm, nj, p0, bm = setup(p, 2, weights.shape[1])
    # raw_margin.setup uses the one-feature margin as a convenient numerical unit.
    # Convert ratios here to the CERTIFIED global ansatz programme, not that unit.
    reference = (3 / 4) * gamma / bm
    z = np.r_[weights.ravel(), 0.]
    assert abs(norm(z)) < 1e-12
    assert abs(ev(z).min() / reference - 1) < 1e-11
    rng = np.random.default_rng(315 + p)
    out = []
    for start in range(starts):
        perturbed = weights + [0, .03, .15, .5][start] * rng.normal(size=weights.shape)
        perturbed *= np.sqrt(p0 / np.sum(perturbed**2))
        z = np.r_[perturbed.ravel(), 0.]
        z[-1] = ev(z).min()
        result = minimize(lambda y: -y[-1], z,
            jac=lambda y: np.r_[np.zeros(y.size-1), -1.], method="SLSQP",
            constraints=[{"type": "eq", "fun": norm, "jac": nj},
                         {"type": "ineq", "fun": con, "jac": jac}],
            options={"maxiter": 600, "ftol": 1e-11})
        ratio = ev(result.x).min() / reference / (np.sum(result.x[:-1]**2)/p0)**1.5
        row = {"p": p, "width": weights.shape[1], "start": start,
               "ratio_to_certified_ansatz": float(ratio), "success": bool(result.success)}
        out.append(row)
        print(row, flush=True)
        if ratio > 1 + 1e-7:
            np.savez(ROOT/f"candidate_p{p}.npz", weights=result.x[:-1].reshape(weights.shape))
    return out


if __name__ == "__main__":
    rows = []
    for p in (3, 5, 7, 11, 13):
        rows.extend(check(p))
        (ROOT / "verification.json").write_text(json.dumps(rows, indent=2) + "\n")
