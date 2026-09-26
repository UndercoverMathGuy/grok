"""Supplementary checks for LOGARITHMIC_SCALING.md; not an empirical proof.

The all-p proof is analytic. This script checks finite inequalities, saved
exact small-prime optimizers, and the concrete random-frequency construction.
"""
import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sp
from scipy.special import i0e, logsumexp

HERE = Path(__file__).resolve().parent


def constants(s):
    a = 3 + s / 2
    c = 0.5 * 9**(1-s)
    C = 1.5 * (2*a)**s
    return a, c, C


def coverage(p, frequencies):
    return 1 - np.cos(2*np.pi*np.arange(p)[:, None]*np.asarray(frequencies)[None, :]/p)


def main():
    rng = np.random.default_rng(20260925)
    result = {"scope": "Numerical consistency checks supplementary to the analytic all-p proof."}

    scalar_cases = 0
    for s in [1.001, 1.1, 1.5, 2, 2.5, 4, 10, 100]:
        a, _, _ = constants(s)
        z = np.r_[0, np.geomspace(1e-12, 1e8, 1200)]
        phi = -np.log(i0e(z))
        rhs = a*np.log1p(z**(1/s))
        assert np.all(phi >= -1e-14)
        assert np.all(phi <= rhs + 1e-13)
        scalar_cases += len(z)
    result["scalar_phi_checks"] = scalar_cases

    finite_cases = 0
    max_violation = 0.0
    for p in [3, 5, 7, 11, 31, 113, 257]:
        for s in [1.1, 1.5, 2, 2.5, 4]:
            a, _, C = constants(s)
            for k in sorted({1, min(3, (p-1)//2), min(12, (p-1)//2), (p-1)//2}):
                frequencies = rng.choice(np.arange(1, (p+1)//2), size=k, replace=False)
                # Include both concentrated and nearly equal allocation patterns.
                for alpha in [0.15, 1.0, 10.0]:
                    w = rng.dirichlet(np.full(k, alpha))
                    y = w**s
                    F = coverage(p, frequencies) @ y
                    m = float(F[1:].min())
                    assert abs(np.sum(y**(1/s))-1) < 1e-12
                    value_bound = C*math.log(p)**(1-s)
                    support_bound = 1.5*math.log(p)/(k*math.expm1(math.log(p)/(2*a*k)))**s
                    assert m <= min(value_bound, support_bound) + 1e-12
                    for t in [0.01, 0.1, 1.0, 10.0, 100.0, (math.log(p)/(2*a))**s]:
                        log_average = float(logsumexp(-t*F)-math.log(p))
                        log_constant_term = float(np.log(i0e(t*y)).sum())
                        entropy_bound = -a*k*math.log1p(t**(1/s)/k)
                        violation = max(log_constant_term-log_average, entropy_bound-log_constant_term)
                        max_violation = max(max_violation, violation)
                        assert violation < 1e-11
                        finite_cases += 1
    result["finite_group_exponential_checks"] = finite_cases
    result["max_floating_inequality_violation"] = max_violation

    # Reuse proved optimizers, without rerunning or trusting a numerical optimizer.
    mp.mp.dps = 70
    exact_checks = []
    s_mp = mp.mpf(3)/2
    for p in [3, 5, 7, 11, 13, 17]:
        data = json.loads((HERE/f"vertices_p{p}.json").read_text())
        root = 2*mp.cos(2*mp.pi/p)
        v = []
        for coeffs in data["amplitudes_descending_polynomial_coefficients"]:
            value = mp.mpf(0)
            for coeff in coeffs:
                value = value*root + mp.mpf(coeff)
            v.append(value)
        cost = sum(x**(1/s_mp) for x in v if x > 0)
        y = [x/cost**s_mp for x in v]
        margin = min(sum(y[j-1]*(1-mp.cos(2*mp.pi*j*d/p)) for j in range(1, len(y)+1)) for d in range(1,p))
        assert abs(margin-cost**(-s_mp)) < mp.mpf("1e-55")
        active = [x for x in y if x > 0]
        k = len(active)
        if k > 1:
            for x in active:
                deleted_rescaled_lower = (margin-2*x)/(1-x**(1/s_mp))**s_mp
                assert deleted_rescaled_lower <= margin + mp.mpf("1e-55")
                assert x**(1/s_mp) >= (margin/2)**(1/(s_mp-1)) - mp.mpf("1e-55")
        assert k <= (2/margin)**(1/(s_mp-1)) + mp.mpf("1e-55")
        a, c, C = constants(1.5)
        assert c*math.log(p)**(-0.5) <= float(margin) <= C*math.log(p)**(-0.5)
        count_lo = math.log(p)/(8*a*a*(1.5/c)**(1/1.5))
        count_hi = 9*4**(1/(1.5-1))*math.log(p)
        assert count_lo <= k <= count_hi
        exact_checks.append(dict(p=p, certified_K=k, gamma=mp.nstr(margin, 30), deletion_condition_passed=True))
    result["saved_exact_optimizer_checks"] = exact_checks

    constructions = []
    for p in [3, 11, 113, 1009, 10007]:
        L = math.ceil(8*math.log(p))
        for attempt in range(100):
            draws = rng.integers(1, (p+1)//2, size=L)
            frequencies, counts = np.unique(draws, return_counts=True)
            cov = coverage(p, frequencies)
            unscaled = cov @ counts
            if unscaled[1:].min() >= L/2:
                break
        else:
            raise AssertionError("No sampled witness; the existence proof does not depend on this finite search.")
        for s in [1.5, 2, 2.5]:
            y = counts.astype(float)*L**(-s)
            budget = float(np.sum(y**(1/s)))
            margin = float((cov@y)[1:].min())
            lower = 0.5*L**(1-s)
            assert budget <= 1+1e-12
            assert margin >= lower-1e-12
            constructions.append(dict(p=p, s=s, draws=L, distinct_frequencies=len(frequencies),
                                      budget=budget, margin=margin, proved_lower_bound=lower,
                                      all_wrong_offsets_checked=p-1))
    result["random_lower_constructions"] = constructions

    nu, b = sp.symbols("nu b", positive=True)
    s = nu/b
    assert sp.simplify(nu/(2*s)-b/2) == 0
    assert sp.simplify((s-1)/s-(1-b/nu)) == 0
    result["penalty_exponent_identities"] = "passed symbolically"
    result["proof_note"] = "LOGARITHMIC_SCALING.md"
    (HERE/"logarithmic_scaling_verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ["random_lower_constructions"]}, indent=2))
    print(f"All {len(constructions)} full-offset lower constructions passed.")


if __name__ == "__main__":
    main()
