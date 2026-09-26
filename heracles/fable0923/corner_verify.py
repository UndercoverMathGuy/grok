"""corner_verify.py -- verify the algebra of the CORNER LEMMA in eta_cert3.py, keeping u an independent symbol.
Delta := 3e^4 - (1 - A eta)^2 (3e^2 + 2eta^2 + 4 eta u + 2u^2),  e = 1 + x,  u^2 = 2x + 2x^2  (u >= 0).
Hand expansion (JOURNAL 2.3):
Delta = 2x + 11x^2 + 12x^3 + 3x^4 - 2eta^2 - 4 eta u + 6A eta + 20A eta x + 14A eta x^2 + 4A eta^3 + 8A eta^2 u
        - 3A^2 eta^2 - 10A^2 eta^2 x - 7A^2 eta^2 x^2 - 2A^2 eta^4 - 4A^2 eta^3 u.
Lower bound after -4 eta u >= -(4 eta^2 + u^2) = -(4 eta^2 + 2x + 2x^2) and dropping nonnegative terms:
Delta >= 9x^2 + 6A eta - eta^2 [6 + 3A^2 + 10A^2 x + 7A^2 x^2 + 2A^2 eta^2 + 4A^2 eta u].
"""
import sympy as sp, numpy as np
x, h, A, u = sp.symbols('x eta A u')
e = 1 + x
Delta = sp.expand(3*e**4 - (1 - A*h)**2 * (3*e**2 + 2*h**2 + 4*h*u + 2*u**2))
# eliminate u^2 only (u kept as a symbol): replace the monomial u**2 by 2x+2x^2
Delta = sp.expand(Delta.subs({u**2: sp.Symbol('U2')}).subs(sp.Symbol('U2'), 2*x + 2*x**2))
hand = (2*x + 11*x**2 + 12*x**3 + 3*x**4 - 2*h**2 - 4*h*u + 6*A*h + 20*A*h*x + 14*A*h*x**2 + 4*A*h**3 + 8*A*h**2*u
        - 3*A**2*h**2 - 10*A**2*h**2*x - 7*A**2*h**2*x**2 - 2*A**2*h**4 - 4*A**2*h**3*u)
print("Delta - hand expansion =", sp.simplify(Delta - hand))
lower = 9*x**2 + 6*A*h - h**2*(6 + 3*A**2 + 10*A**2*x + 7*A**2*x**2 + 2*A**2*h**2 + 4*A**2*h*u)
# Delta - lower must equal  (u^2 + 4 eta^2 - 4 eta u) + (dropped nonneg terms)  where u^2 = 2x+2x^2:
resid = sp.expand(hand - lower - ((2*x + 2*x**2) + 4*h**2 - 4*h*u))
print("Delta - lower - (u - 2 eta)^2 =", sp.factor(resid), " (every term should have a nonnegative coefficient)")
print("nonnegative coefficients:", all(c > 0 for c in sp.Poly(resid, x, h, u, A).coeffs()))
# numeric spot check
rng = np.random.default_rng(0)
f = sp.lambdify((x, h, A), Delta.subs(u, sp.sqrt(2*x + 2*x**2))); g = sp.lambdify((x, h, A), lower.subs(u, sp.sqrt(2*x + 2*x**2)))
X = rng.random(200000)*0.5; H = rng.random(200000)*0.5; AA = rng.random(200000)*0.5
print("min(Delta - lower) over random points:", float((f(X, H, AA) - g(X, H, AA)).min()))
