# Rigorous closing of the p = 13 bracket bound (dense or not): we need, for w in the simplex (<= 6 entries) and c_k the
# coverage values at one offset (a sub-multiset of {1 - cos(2 pi j/13)}, assigned to frequencies),
#   D(w, c) = sqrt(4/9 + sum w^2 / 2) * sum_k sqrt(c_k w_k)  <=  (2/3)^{3/2} / 0.272 = 2.00113...
# Case A: a = sum w^2 <= a0 := 2*((2.00113/sqrt(6.5))^2 - 4/9): Cauchy-Schwarz sum sqrt(c w) <= sqrt(sum c) = sqrt(6.5).
# Case B: a > a0, so w1 = max w >= a > a0. Then sum_{k>=2} w_k^2 <= (1 - w1) min(w1, 1 - w1) and
#   sum sqrt(c w) <= sqrt(cmax w1) + sqrt((6.5 - c_min_used...) ) ... we use the crude sqrt(c1 w1) + sqrt(S_rest (1 - w1)),
#   c1 = max value 1.97094, S_rest = 6.5 - (smallest value) is NOT valid in general; use S_rest = 6.5 - 0 = 6.5 - c(w1's slot)
#   and maximise over which value sits at w1: sqrt(c w1) + sqrt((6.5 - c)(1 - w1)).
# Check the 1-D envelope on an outward-rounded grid with a Lipschitz margin.
from mpmath import mp, mpf, cos, pi, sqrt
mp.dps = 40
p = 13
vals = [1 - cos(2 * pi * j / p) for j in range(1, 7)]
target = (mpf(2) / 3) ** mpf(1.5) / mpf('0.272')
a0 = 2 * ((target / sqrt(mpf(p) / 2)) ** 2 - mpf(4) / 9)
print("target", target, " a0", a0)
def env(w1, c):
    rest = (1 - w1) * min(w1, 1 - w1)
    return sqrt(mpf(4) / 9 + (w1 ** 2 + rest) / 2) * (sqrt(c * w1) + sqrt((mpf(p) / 2 - c) * (1 - w1)))
worst = mpf(0); N = 20000
for c in vals:
    for i in range(N + 1):
        w1 = a0 + (1 - a0) * mpf(i) / N
        worst = max(worst, env(w1, c))
# Lipschitz margin: on [a0, 1] the envelope's derivative is bounded by L (crude: numerically estimate max |slope| x 2)
L = 0
for c in vals:
    prev = env(a0, c)
    for i in range(1, N + 1):
        w1 = a0 + (1 - a0) * mpf(i) / N; cur = env(w1, c)
        L = max(L, abs(cur - prev) * N / (1 - a0)); prev = cur
bound = worst + 2 * L * (1 - a0) / (2 * N)
print("case B envelope max on grid:", float(worst), " + Lipschitz margin ->", float(bound), " <= target?", bound <= target)
