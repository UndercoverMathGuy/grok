"""Independent rational vertex certificate for the degree-four S5 H ceiling.

Characters are reconstructed from the natural permutation action. Feasibility,
vertex enumeration, and comparisons use rational arithmetic throughout.
Square roots are enclosed by integer square roots with outward rational bounds.
"""
import itertools
import json
from fractions import Fraction as Q
from math import isqrt
from pathlib import Path

HERE = Path(__file__).resolve().parent


def solve(A, b):
    n = len(b)
    M = [[Q(x) for x in row] + [Q(y)] for row, y in zip(A, b)]
    for j in range(n):
        pivot = next((i for i in range(j, n) if M[i][j]), None)
        if pivot is None:
            return None
        M[j], M[pivot] = M[pivot], M[j]
        scale = M[j][j]
        M[j] = [x / scale for x in M[j]]
        for i in range(n):
            if i != j:
                scale = M[i][j]
                M[i] = [x - scale*y for x, y in zip(M[i], M[j])]
    return tuple(row[-1] for row in M)


def sqrt_interval(x):
    scale = 10**15
    lo = isqrt(x.numerator * scale**2 // x.denominator)
    assert Q(lo, scale)**2 <= x < Q(lo + 1, scale)**2
    return Q(lo, scale), Q(lo + 1, scale)


def main():
    names = ["sign", "std", "std_sign", "Sym2std-trivial-std", "sign*(Sym2std-trivial-std)", "wedge2std"]
    dims = [1, 4, 4, 5, 5, 6]
    coverages = set()
    for p in itertools.permutations(range(5)):
        if p == tuple(range(5)):
            continue
        sign = (-1)**sum(p[i] > p[j] for i in range(5) for j in range(i+1, 5))
        chi = sum(p[i] == i for i in range(5)) - 1
        chi2 = sum(p[p[i]] == i for i in range(5)) - 1
        five = (chi*chi + chi2)//2 - 1 - chi
        six = (chi*chi - chi2)//2
        chars = [sign, chi, sign*chi, five, sign*five, six]
        coverages.add(tuple(d-c for d, c in zip(dims, chars)))
    coverage = sorted(coverages)
    assert len(coverage) == 6
    constraints = coverage + [tuple(int(i == j) for j in range(6)) for i in range(6)]
    rhs = [1]*6 + [0]*6
    vertices = set()
    for active in itertools.combinations(range(12), 6):
        y = solve([constraints[i] for i in active], [rhs[i] for i in active])
        if y is None or any(t < 0 for t in y):
            continue
        if any(sum(c*t for c, t in zip(row, y)) < 1 for row in coverage):
            continue
        vertices.add(y)
    winner = (Q(0), Q(0), Q(1, 3), Q(0), Q(0), Q(0))
    assert winner in vertices
    prices = [d**3 for d in dims]  # K2: d^(nu-q)*d^(q/2), nu=4, q=2.
    costs = {}
    for y in vertices:
        intervals = [sqrt_interval(t) for t in y]
        costs[y] = tuple(sum(price*bounds[j] for price, bounds in zip(prices, intervals)) for j in (0, 1))
    upper = costs[winner][1]
    other_slacks = [lo - upper for y, (lo, hi) in costs.items() if y != winner]
    assert min(other_slacks) > 0
    result = dict(
        irreps=names, dimensions=dims, coverage_rows=coverage,
        vertex_count=len(vertices), unique_winner=[str(x) for x in winner],
        exact_winning_cost="64/sqrt(3)", exact_ceiling="3/30720^2",
        smallest_certified_cost_slack=str(min(other_slacks)),
        vertices=[dict(y=[str(x) for x in y], cost_interval=[str(x) for x in costs[y]]) for y in sorted(vertices)],
    )
    (HERE / "s5_ceiling_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Exact certificate: {len(vertices)} vertices; unique winner std_sign; all other cost intervals strictly higher.")
    print(f"Minimum rational cost slack: {min(other_slacks)}")


if __name__ == "__main__":
    main()
