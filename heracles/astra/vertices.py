"""Complete algebraic vertex enumeration for the ansatz, not the full network.

Run: UV_CACHE_DIR=/private/tmp/heracles-uv-cache uv run --offline python
     heracles/astra/vertices.py --primes 3 5 7 11 13
Exact rational arithmetic in Q(2*cos(2*pi/p)) decides singularity and equality.
Directed interval evaluation decides signs and separates objective classes.
No floating-point candidate screening is used.
"""
import argparse
import itertools
import json
from pathlib import Path

import mpmath as mp
import sympy as sp


def solve(matrix, rhs, zero):
    a = [list(row) + [v] for row, v in zip(matrix, rhs)]
    n = len(a)
    for j in range(n):
        pivot = next((i for i in range(j, n) if a[i][j] != zero), None)
        if pivot is None:
            return None
        a[j], a[pivot] = a[pivot], a[j]
        z = a[j][j]
        a[j] = [v / z for v in a[j]]
        for i in range(n):
            if i != j and a[i][j] != zero:
                z = a[i][j]
                a[i] = [u - z * v for u, v in zip(a[i], a[j])]
    return [row[-1] for row in a]


def enumerate_prime(p, numerator=2, denominator=3):
    assert sp.isprime(p) and p > 2
    h = (p - 1) // 2
    field = sp.QQ.algebraic_field(2 * sp.cos(2 * sp.pi / p))
    zero, one = field.zero, field.one
    alpha = field.from_sympy(2 * sp.cos(2 * sp.pi / p))
    # t[j] = 2*cos(2*pi*j/p), by the Chebyshev recurrence.
    t = [2 * one, alpha]
    for j in range(2, h + 1):
        t.append(alpha * t[-1] - t[-2])
    fold = lambda j: min(j % p, (-j) % p)
    a = [[one - t[fold(d * k)] / 2 for k in range(1, h + 1)]
         for d in range(1, h + 1)]
    mp.iv.dps = 80
    root = 2 * mp.iv.cos(2 * mp.iv.pi / p)

    def interval(v):
        out = mp.iv.mpf(0)
        for c in v.to_list():
            out = out * root + mp.iv.mpf(int(c.numerator)) / int(c.denominator)
        return out

    def sign(v):
        if v == zero:
            return 0
        z = interval(v)
        if z.a > 0:
            return 1
        if z.b < 0:
            return -1
        raise ArithmeticError("Unresolved algebraic sign: increase interval precision")

    def key(v):
        return tuple(str(c) for c in v.to_list())

    tested = singular = feasible = 0
    classes = {}
    vertices = set()
    for k in range(1, h + 1):
        # Frequency dilation maps every nonempty support to one containing 1.
        for tail in itertools.combinations(range(1, h), k - 1):
            support = (0,) + tail
            for active in itertools.combinations(range(h), k):
                tested += 1
                v = solve([[a[d][j] for j in support] for d in active], [one] * k, zero)
                if v is None:
                    singular += 1
                    continue
                if any(sign(x) < 0 for x in v):
                    continue
                if any(sign(sum((a[d][j] * x for j, x in zip(support, v)), zero) - one) < 0
                       for d in range(h)):
                    continue
                feasible += 1
                full = [zero] * h
                for j, x in zip(support, v):
                    full[j] = x
                vertex_key = tuple(key(x) for x in full)
                vertices.add(vertex_key)
                positives = [x for x in full if x != zero]
                class_key = tuple(sorted(key(x) for x in positives))
                if class_key not in classes:
                    objective = sum((interval(x) ** (mp.iv.mpf(numerator) / denominator)
                                     for x in positives), mp.iv.mpf(0))
                    classes[class_key] = (objective, full)
    winner = min(classes, key=lambda k: float(classes[k][0].a))
    cost, full = classes[winner]
    # This strict interval separation is the objective comparison certificate.
    assert all(cost.b < obj.a for k, (obj, _) in classes.items() if k != winner)
    gamma = cost ** (-mp.iv.mpf(denominator) / numerator)
    result = {
        "p": p, "r": f"{numerator}/{denominator}", "interval_precision": mp.iv.dps,
        "minimal_polynomial": str(field.ext.minpoly.as_expr()),
        "root": "2*cos(2*pi/p)", "candidate_systems": tested,
        "singular_systems": singular, "feasible_systems": feasible,
        "distinct_enumerated_vertices": len(vertices), "amplitude_multiset_classes": len(classes),
        "support": [j + 1 for j, v in enumerate(full) if v != zero],
        "K": sum(v != zero for v in full),
        "amplitudes_descending_polynomial_coefficients": [key(v) for v in full],
        "cost_interval": str(cost), "gamma_interval": str(gamma),
        "strict_separation_from_other_classes": True,
        "objective_class_intervals": [str(obj) for obj, _ in classes.values()],
        "scope": "Complete ansatz optimum, all supports, modulo frequency dilation. Not unrestricted network optimality.",
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--primes", type=int, nargs="+", default=[3, 5, 7, 11, 13])
    args = parser.parse_args()
    for p in args.primes:
        result = enumerate_prime(p)
        Path(__file__).with_name(f"vertices_p{p}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: result[k] for k in ("p", "K", "support", "gamma_interval", "candidate_systems")}), flush=True)
