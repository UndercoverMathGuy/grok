# Lemma (monotonicity with a.e. hypothesis): sanity illustrations.
# (1) necessity of the bound D^+L <= C: the Cantor function has derivative 0 off a null set yet increases by 1;
#     its difference quotients at Cantor points blow up like (3/2)^k.
# (2) the lemma's mechanism: with phi <= C and phi <= 0 off an open set U of measure eps/C, L(c)-L(a) <= eps.
import numpy as np
def cantor(x, depth=40):
    y = 0.0; f = 0.5
    for _ in range(depth):
        if x < 1/3: x = 3*x
        elif x > 2/3: y += f; x = 3*x - 2
        else: return y + f
        f /= 2
    return y
xs = np.linspace(0, 1, 200001); c = np.array([cantor(x) for x in xs[::100]])
print(f"(1) Cantor: c(1)-c(0) = {cantor(1-1e-15)-cantor(0):.6f}; max difference quotient at 0 over h=3^-k: "
      f"{[round((cantor(3.0**-k)-cantor(0))*3**k,3) for k in (2,6,10,14)]} (grows like (3/2)^k: D^+ = +inf)")
# (2) random L with D^+L <= C on U (union of short intervals), and D^+L <= 0 elsewhere
rng = np.random.default_rng(0); C = 3.0; eps = 1e-2
cent = rng.uniform(0, 1, 20); half = eps / C / 40
s = np.linspace(0, 1, 400001); slope = np.where(np.any(np.abs(s[:, None] - cent[None, :]) < half, 1), C, -rng.uniform(0, 1))
L = np.concatenate([[0], np.cumsum(slope[:-1] * np.diff(s))])
print(f"(2) measure(U) = {np.mean(slope == C):.2e} <= eps/C = {eps/C:.2e};  max_{{a<c}} L(c)-L(a) = {np.max(L - np.minimum.accumulate(L)):.2e} <= eps = {eps}")
