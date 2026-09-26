"""Exact identity checks for ENDPOINT_NEIGHBORHOOD.md; no optimization/search.

The theorem is analytic. Finite examples below independently check Fourier
normalizations, block cancellation, the frame identity, and the comparison
algebra. They do not substitute for the all-parameter proof.
"""

import itertools
import json
from pathlib import Path

import sympy as s


def root_field(order):
    x = s.Symbol("x")
    modulus = s.Poly(s.cyclotomic_poly(order, x), x, domain=s.QQ)

    def reduce(poly):
        return s.rem(s.Poly(poly, x, domain=s.QQ), modulus).as_expr()

    powers = [reduce(x**j) for j in range(order)]
    return reduce, powers


record = {"arithmetic": "exact rational / cyclotomic / symbolic", "checks": {}}

# All block sign patterns, including even nu: the phase is pi*l/nu,
# so flipping |T| input signs contributes exp(2*pi*i*l*|T|/nu).
block_counts = {}
for nu in range(3, 13):
    reduce, powers = root_field(nu)
    for pattern in itertools.product((0, 1), repeat=nu - 1):
        flips = sum(pattern)
        result = reduce(sum(powers[(ell * flips) % nu] for ell in range(nu)))
        assert result == (nu if flips == 0 else 0)
    # Running products form the discrete Fourier matrix at these phases.
    for j in range(nu):
        for k in range(nu):
            assert reduce(sum(powers[(ell * (j-k)) % nu] for ell in range(nu))) == (
                nu if j == k else 0
            )
    block_counts[str(nu)] = 2 ** (nu - 1)
record["checks"]["all_sign_patterns_and_chain_frames"] = block_counts

# Average every complex Fourier basis tensor on every row and wrong offset.
# Independence of input coordinates makes the row sum a product of finite sums.
dual_counts = {}
for p, nu in ((3, 3), (5, 3), (7, 3), (3, 4), (5, 4)):
    reduce, powers = root_field(p)
    sums = [reduce(sum(powers[(freq * x) % p] for x in range(p))) for freq in range(p)]
    checked = 0
    for frequencies in itertools.product(range(p), repeat=nu):
        inputs, output = frequencies[:-1], frequencies[-1]
        row_sum = s.prod(sums[(freq + output) % p] for freq in inputs)
        wrong_sum = reduce(sum(1 - powers[(output * offset) % p] for offset in range(1, p)))
        average = reduce(row_sum * wrong_sum / (s.Integer(p)**(nu - 1) * (p - 1)))
        expected = s.Rational(p, p - 1) if output and all(
            (freq + output) % p == 0 for freq in inputs
        ) else 0
        assert average == expected
        checked += 1
    dual_counts[f"p={p},nu={nu}"] = checked
record["checks"]["average_gap_on_entire_fourier_basis"] = dual_counts

# Full row/class tables from the dense Fourier coefficients, exactly in Q(zeta_p).
table_counts = {}
for p, nu in ((3, 3), (5, 3), (7, 3), (3, 4), (5, 4)):
    reduce, powers = root_field(p)
    by_offset = [reduce(sum(powers[(k*d) % p] for k in range(1, p))/p) for d in range(p)]
    checked = 0
    for inputs in itertools.product(range(p), repeat=nu - 1):
        correct = sum(inputs) % p
        for output in range(p):
            z = by_offset[(output - correct) % p]
            assert z == (s.Integer(output == correct) - s.Rational(1, p))
            if output != correct:
                assert by_offset[0] - z == 1
            checked += 1
    table_counts[f"p={p},nu={nu}"] = checked
record["checks"]["full_dense_logit_tables"] = table_counts

# Square frame inversion: deliberately non-orthogonal chain vectors, arbitrary
# positive weights. This checks the identity used away from the exact endpoint.
phases = [s.Integer(1), s.I, s.Rational(3, 5) + s.I*s.Rational(4, 5), -s.I]
frame_records = []
for nu in (3, 4):
    columns = [s.Matrix([phase**j for j in range(nu)]) for phase in phases[:nu]]
    frame = s.Matrix.hstack(*columns)
    assert s.simplify(frame.det()) != 0
    weights = [s.Rational(10+i, 11) for i in range(nu)]
    matrix = frame * s.diag(*weights) * frame.conjugate().T
    inverse = matrix.inv()
    for column, weight in zip(columns, weights):
        assert s.simplify((column.conjugate().T * inverse * column)[0] - 1/weight) == 0
        assert s.simplify((column.conjugate().T * column)[0] - nu) == 0
    frame_records.append({"nu": nu, "nonorthogonal": True, "inverse_identity": True})
record["checks"]["frame_inversion_away_from_endpoint"] = frame_records

# Normalization and comparison algebra with unrestricted symbolic parameters.
p, nu, r = s.symbols("p nu r", positive=True)
h = (p-1)/2
a0 = (2*p)**(nu/2)/(p*nu)
beta = (2*p)**(-nu/2)
assert s.simplify(a0*beta - 1/(p*nu)) == 0
assert s.simplify(a0*nu*h - (2*p)**(nu/2)*(p-1)/(2*p)) == 0

E, t, eps, C = s.symbols("E t eps C", nonnegative=True)
comparison = (1-eps)*(E-t) - C*eps*(E+t*t) + t  # t^r >= t for t<=1
expected = (1-(C+1)*eps)*E + eps*t - C*eps*t*t
assert s.expand(comparison-expected) == 0

# For delta<=1/100, inconsistent dominant frequencies contradict s>=1-delta.
delta = s.Rational(1, 100)
assert (1-delta-2*s.sqrt(2*delta)-2*delta).is_positive

# The second exponent derivative extends continuously by zero at b=0.
b = s.Symbol("b", positive=True)
assert s.limit(s.sqrt(b)*s.log(b)**2, b, 0, dir="+") == 0
assert s.simplify(s.diff(b**r, r, 2) - b**r*s.log(b)**2) == 0
assert s.simplify(s.diff(b**r, b, 2) + r*(1-r)*b**(r-2)) == 0

record["checks"]["normalization_entropy_and_local_comparison"] = True
record["scope"] = (
    "Exact checks of the analytic proof's identities, not empirical evidence "
    "or a finite-prime proof of the endpoint-neighborhood theorem."
)
out = Path(__file__).with_name("endpoint_verification.json")
out.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
