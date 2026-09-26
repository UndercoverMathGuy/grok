"""Exact checks for the explicit endpoint proof; no optimizer or sampling."""

import itertools
import json
from pathlib import Path

import sympy as s


record = {"arithmetic": "exact", "checks": {}}

# All constants and the optimized Young identity.
A, degree = s.symbols("A degree", positive=True)
c = (A + s.sqrt(A*A + 32*degree))/2
eta = c/A - 1
assert s.simplify(c*c - A*c - 8*degree) == 0
assert s.simplify(A*A*eta*(1+eta) - 8*degree) == 0
assert s.simplify(c*c/2 - 4*degree*(1+1/eta)) == 0

z = s.symbols("z", nonnegative=True)
assert s.expand((2+z)*(1+z)/2 - (z+z*z/2)) == 1+z/2

# Normalize the global residual bound with B=N0+E, N0=nu*h.
nu, h, x, cc = s.symbols("nu h x cc", positive=True)
N = nu*h
E = N*x
B = N+E
residual = (cc/nu)*B*E + (cc*cc/(2*nu))*h*E*E
assert s.simplify(residual/N - (cc*h*x + (cc*h+cc*cc*h*h/2)*x*x)) == 0
record["checks"]["optimized_young_and_scalar_comparison"] = True

# Depth-uniform simpler interval: c_nu<=3*nu and the gap increases with nu.
nn, hh = s.symbols("nn hh", positive=True)
assert s.factor(nn/(1+3*nn*hh)-3/(1+9*hh)) == (nn-3)/((9*hh+1)*(3*hh*nn+1))
assert s.expand((2*nn)**2-(nn*nn+8*nn)-nn*(3*nn-8)) == 0
record["checks"]["depth_uniform_interval_algebra"] = True

# Rational simplifications of the thresholds.
thresholds = []
for degree_value, safe_constant in ((3, 9), (4, 11), (5, 14)):
    av = s.Integer(2*degree_value)
    # c is the positive root of c^2-A*c-8*nu=0.
    assert safe_constant**2-av*safe_constant-8*degree_value > 0
    for p in (5, 7, 11, 59):
        half = (p-1)//2
        q = s.Rational(degree_value*safe_constant*half, 1+safe_constant*half)
        thresholds.append({"p": p, "nu": degree_value, "safe_c": safe_constant,
                           "q_lower_exact": str(q), "q_lower_decimal": str(s.N(q, 14))})
record["checks"]["rational_thresholds"] = thresholds

# Fourier masks used by the chain map are distinct and disjoint from their
# conjugates; verify the exact multiplicity accounting as well.
mask_checks = []
for p in (3, 5, 7):
    for degree_value in range(3, 8):
        n = degree_value-1
        masks = set()
        for k in range(1, (p-1)//2+1):
            clean = tuple([-k % p]*n + [k])
            assert clean not in masks
            masks.add(clean)
            for left in range(n):
                for right in range(left, n):
                    mask = tuple([k if left <= j <= right else -k % p for j in range(n)] + [k])
                    assert mask not in masks
                    masks.add(mask)
        assert not masks.intersection({tuple(-v % p for v in mask) for mask in masks})
        clean_ratio = s.Rational(2**(degree_value-1), degree_value)
        junk_ratio = s.Rational(2**degree_value, degree_value**2)
        assert junk_ratio <= clean_ratio
        mask_checks.append({"p": p, "nu": degree_value, "selected_masks": len(masks),
                            "map_norm_squared": str(clean_ratio)})
record["checks"]["fourier_map_masks_and_exact_norm"] = mask_checks

# Matrix projection estimate on deliberately mixed-frequency/DC unit slots. All
# complex coordinates and their magnitudes are rational, so this is exact.
phases = [s.Integer(1), s.I, (3+4*s.I)/5, (-5+12*s.I)/13]
slot_types = [(s.Rational(3, 5), s.Rational(4, 5), s.Integer(0)),
              (s.Rational(2, 3), s.Rational(2, 3), s.Rational(1, 3)),
              (s.Rational(5, 13), s.Rational(12, 13), s.Integer(0))]
projection_count = 0
for degree_value in range(3, 9):
    for shift in range(12):
        coords = []
        for j in range(degree_value):
            magnitudes = slot_types[(shift+j) % len(slot_types)]
            if j == 0:  # centered readout
                magnitudes = slot_types[shift % 2 * 2]
            assert sum(v*v for v in magnitudes) == 1
            coords.append([magnitudes[k]*phases[(shift+2*j+k) % len(phases)] for k in (0, 1)])
        clean = [s.simplify(coords[0][k]*s.prod(s.conjugate(coords[j][k])
                                             for j in range(1, degree_value))) for k in (0, 1)]
        radii = [s.simplify(s.Abs(v)) for v in clean]
        k = max(range(2), key=lambda index: radii[index])
        deficit = s.simplify(1-sum(s.re(v) for v in clean))
        matrix_norm_sq = s.simplify(sum(radii[j]**2-s.im(clean[j])**2/degree_value for j in (0, 1)))
        matrix_distance_sq = s.simplify(1+matrix_norm_sq-2*s.re(clean[k]))
        assert deficit >= 0
        assert s.simplify(2*deficit-matrix_distance_sq) >= 0
        projection_count += 1
record["checks"]["exact_mixed_and_dc_matrix_projection_examples"] = projection_count

# This factorization proves the scalar bound for all dimensions nu>=3,
# after the monotonicity reduction to nu=3 in the proof.
tt = s.symbols("tt", real=True)
assert s.expand(1-tt**4-2*tt*(1-tt**2)-(1-tt)**3*(1+tt)) == 0
record["checks"]["all_parameter_projection_scalar_factorization"] = True

# Full feasible centered tensors with rational nonnegative gap excesses.
# Check every selected Fourier coefficient against the 2E/nu entry bound.
root = s.Symbol("root")
residual_counts = []
for p, degree_value in ((3, 3), (5, 3), (3, 4)):
    poly = s.Poly(s.cyclotomic_poly(p, root), root, domain=s.QQ)

    def reduce(value):
        return s.rem(s.Poly(value, root, domain=s.QQ), poly).as_expr()

    powers = [reduce(root**j) for j in range(p)]
    n = degree_value-1
    half = (p-1)//2
    excess = {}
    for inputs in itertools.product(range(p), repeat=n):
        y = sum(inputs) % p
        for output in range(p):
            excess[inputs+(output,)] = (s.Integer(0) if output == y else
                s.Rational((sum((j+1)*v for j, v in enumerate(inputs)) + 2*output) % 7, 17))
    total = sum(excess.values())
    endpoint_excess = s.Rational(degree_value*half, p**n*(p-1))*total
    bound = 2*endpoint_excess/degree_value
    for k in range(1, half+1):
        masks = [tuple([-k % p]*n + [k])]
        masks += [tuple([k if left <= j <= right else -k % p for j in range(n)] + [k])
                  for left in range(n) for right in range(left, n)]
        for mask in masks:
            # Q on (Z-Z0)/a0 is p times this nonzero-readout Fourier coefficient.
            coefficient = reduce(-sum(value*powers[-sum(a*b for a, b in zip(mask, index)) % p]
                                      for index, value in excess.items()) / p**n)
            conjugate = reduce(coefficient.subs(root, powers[p-1]))
            modulus_sq = reduce(coefficient*conjugate)
            # Exact algebraic sign, not a floating tolerance.
            gap = reduce(bound*bound-modulus_sq)
            algebraic_gap = s.simplify(gap.subs(root, s.exp(2*s.pi*s.I/p)).expand(complex=True))
            assert algebraic_gap.is_nonnegative
    residual_counts.append({"p": p, "nu": degree_value, "table_entries": p**degree_value,
                            "gap_residual_bound": True})
record["checks"]["full_feasible_tensor_residuals"] = residual_counts

record["scope"] = "Exact checks of proof identities; no empirical optimization or claim of finite-example completeness."
Path(__file__).with_name("explicit_endpoint_verification.json").write_text(json.dumps(record, indent=2)+"\n")
print(json.dumps(record, indent=2))
