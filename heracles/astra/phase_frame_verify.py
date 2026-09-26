"""Exact verification for PHASE_FRAME_EXTENSION.md. No optimizer search."""

import itertools
import json
from pathlib import Path

import sympy as s


record = {"arithmetic": "exact symbolic / rational / algebraic", "checks": {}}

# Geometric phase eigenvectors: exp(i*alpha/nu) is represented by a formal z.
z = s.symbols("z", nonzero=True)
phase_records = []
for nu in range(3, 10):
    cosine = (z**nu+z**(-nu))/2
    H = s.Matrix(nu, nu, lambda a, b: cosine if a == b else z**(nu if a > b else -nu))
    vector = s.Matrix([z**(2*j) for j in range(nu)])
    eigenvalue = (z**nu-z**(-nu))*(z+1/z)/(2*(z-1/z))
    assert all(s.cancel(value) == 0 for value in H*vector-eigenvalue*vector)
    phase_records.append({"nu": nu, "geometric_eigenvector": True})
record["checks"]["phase_frame_eigenvectors"] = phase_records

# Exact cubic phase identities and the polynomial that bounds its remainder.
u, d, x = s.symbols("u d x", real=True)
ca = 4*(1-u)**3-3*(1-u)
assert s.expand(1-ca-u*(3-2*u)**2) == 0
lam = (ca+2*(1-u))/3
distance = 2-(1-ca**2)/3-2*lam
assert s.expand(distance-((1-ca)**2+4*u)/3) == 0
phase_remainder = 5*(3-2*u)**4-32*(3-u)
assert s.expand(phase_remainder-(1-2*u)*(309-430*u+220*u*u-40*u**3)) == 0
assert s.Rational(1, 3)+s.Rational(4, 3)*s.Rational(5, 72) == s.Rational(23, 54)
assert s.Rational(23, 54) < s.Rational(4, 9)
record["checks"]["cubic_phase_remainder"] = True

# Exact norm of an individual gap generator and the depth-uniform cutoff.
h, nu = s.symbols("h nu", positive=True)
p = 2*h+1
gap_norm = 4/nu*((p-2)/4+h*(nu-1))
assert s.simplify(gap_norm-(4*h-p/nu)) == 0
N = nu*h
uniform_difference = s.expand((1+3*N)**2-(1+8*(1+gap_norm)*N))
assert s.simplify(uniform_difference-N*(-2+9*N-8*gap_norm)) == 0
assert s.simplify((-2+9*N-8*gap_norm).subs(nu, 3)-(h+2)/3) == 0
record["checks"]["gap_norm_and_uniform_cutoff"] = True

# Complete p=5 q=5/2 scalar certificate.
P = s.expand(9126*((1+x)**7-1-2*x)-(667*x+4)**2)
assert P == (9126*x**7+63882*x**6+191646*x**5+319410*x**4
             +319410*x**3-253243*x**2+40294*x-16)
R = 40294-253243*x+319410*x*x+319410*x**3
positive_decomposition = (319410*x*(x-s.Rational(7, 25))**2
    +s.Rational(2491398, 5)*(x-s.Rational(7, 25))**2
    +s.Rational(93979, 125)*x+s.Rational(3840248, 3125))
assert s.expand(R-positive_decomposition) == 0
assert s.Rational(3840248, 3125) > 1200
assert 40294-s.Rational(253243, 50) > 35000
assert -16+s.Rational(140000, 503) > 0
assert s.Rational(4, 507)/(1-s.Rational(4, 507)) == s.Rational(4, 503)
assert s.Rational(17, 6)**2 > s.Rational(19, 3)
assert s.Rational(4, 27)/(4*s.Rational(13, 6)**2) == s.Rational(4, 507)
assert s.simplify(s.Rational(17, 6)*6*x+s.Rational(4, 27)*6*(1+x)/(4*s.Rational(13, 6))
                 -(667*x+4)/39) == 0
record["checks"]["p5_q_five_halves_positive_polynomial_certificate"] = True

# The exact matrix-only obstruction, including the complete gap table.
root5 = s.sqrt(5)
p = 5
excess = {((3, 3), 2): 5*root5, ((2, 2), 3): 5*root5}
tensor = {}
for inputs in itertools.product(range(p), repeat=2):
    correct = sum(inputs) % p
    row_mean = s.Rational(1, 5)*sum(excess.get((inputs, c), 0) for c in range(p))
    for c in range(p):
        tensor[inputs+(c,)] = s.Integer(c == correct)-s.Rational(1, 5)-excess.get((inputs, c), 0)+row_mean
    assert s.simplify(sum(tensor[inputs+(c,)] for c in range(p))) == 0
    for c in range(p):
        if c != correct:
            assert s.simplify(tensor[inputs+(correct,)]-tensor[inputs+(c,)]-1-excess.get((inputs, c), 0)) == 0
for index, value in tensor.items():
    assert s.simplify(value-tensor[tuple(-v % p for v in index)]) == 0

c = (root5-1)/4
dd = -(root5+1)/4
scale = 2/root5
J = s.ones(3)
H1 = (c-dd)*s.eye(3)+dd*J
H2 = (dd-c)*s.eye(3)+c*J
X1, X2 = s.eye(3)-scale*H1, s.eye(3)-scale*H2
uu = (1+3/root5)/2
assert all(s.simplify(v) == 0 for v in X1-(1+uu)*J/3)
assert all(s.simplify(v) == 0 for v in X2-(2*s.eye(3)+(uu-2)*J/3))

# Direct phase extraction from the two nonzero excesses, for every Q entry.
for k, target in ((1, H1), (2, H2)):
    for row in range(3):
        for col in range(3):
            subset = list(range(min(row, col), max(row, col)))
            residue = (1+2*sum((3, 3)[j] for j in subset)) % 5
            expected = s.cos(2*s.pi*k*(1 if row == col else residue)/5)
            assert s.simplify(expected-target[row, col]) == 0

# Certified rational upper bound on the 2/3 spectral power sum.
assert s.Rational(9, 5) < s.Rational(671, 500)**2  # uu < 1171/1000
for value_bound, power_bound in ((s.Rational(2171, 1000), s.Rational(42, 25)),
                                (s.Rational(1171, 1000), s.Rational(28, 25)),
                                (s.Integer(2), s.Rational(159, 100))):
    assert value_bound**2 < power_bound**3
upper = s.Rational(42, 25)+s.Rational(28, 25)+2*s.Rational(159, 100)
assert upper == s.Rational(299, 50) and upper < 6

# The inexpensive matching network violates this actual margin constraint.
cos1_sq = (3-root5)/8
cos2_sq = (3+root5)/8
A1 = (5+root5)/4
A2 = (5-root5)/4
gap = s.Rational(8, 15)*((1+uu)*cos1_sq*A1+(uu-2)*cos2_sq*A2+s.Rational(3, 2)*A2)
assert s.simplify(gap-(1-1/root5)) == 0
record["checks"]["matrix_only_obstruction"] = {
    "full_centered_table_entries": 125,
    "all_100_wrong_class_gaps_verified": True,
    "reflection_symmetry_and_real_fourier_coefficients": True,
    "all_retained_matrix_entries_verified": True,
    "psd_eigenvalues": ["0", "0", "1+u", "2", "2", "u"],
    "u": "(1+3/sqrt(5))/2",
    "spectral_power_sum_strict_upper": "299/50 < 6",
    "cheap_matching_network_violated_gap": "1-1/sqrt(5) < 1",
    "not_a_counterexample_to_network_global_optimality": True,
}
record["scope"] = "Exact verification of analytic proof identities and a relaxation obstruction; the full q=2 network theorem remains open."
Path(__file__).with_name("phase_frame_verification.json").write_text(json.dumps(record, indent=2)+"\n")
print(json.dumps(record, indent=2))
