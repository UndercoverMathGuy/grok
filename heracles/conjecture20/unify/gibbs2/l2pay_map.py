# 2-D reduced form of the sharp L2 payment at s = 3/2 (b = 2).
#   region: 0 <= delta <= 1, 0 <= beta <= sqrt(1+delta);  P = beta^2 + delta^2;  q* = max(P-1, -beta^2 delta);
#   nu = 1 + roots(t^3 - P t - q*);  RHS = M_{2/3}(nu) = (mean nu^{2/3})^{3/2};
#   LHS = min_{l in [0,1]} sqrt(1 + c^2 l^2 - 2 c l delta) - c beta sqrt(1 - l^2),   c = (sqrt3 - 1)/3.
#   Claim: LHS <= RHS on the region (equality at (0,0) and (sqrt2, 1)).
import numpy as np
c = (np.sqrt(3) - 1)/3; r = 2/3
def rhs(beta, d):
    P = beta*beta + d*d; q = max(P - 1, -beta*beta*d)
    t = np.roots([1, 0, -P, -q]).real; nu = np.clip(1 + t, 0, None)
    return (np.mean(nu**r))**1.5
ls = np.linspace(0, 1, 2001)
def lhs(beta, d):
    return np.min(np.sqrt(1 + c*c*ls*ls - 2*c*ls*d) - c*beta*np.sqrt(1 - ls*ls))
worst = []
N = 300
for i in range(N + 1):
    d = i/N
    for j in range(N + 1):
        beta = np.sqrt(1 + d)*j/N
        m = rhs(beta, d) - lhs(beta, d)
        dist0 = np.hypot(beta, d); dist1 = np.hypot(np.sqrt(2) - beta, 1 - d)
        worst.append((m, beta, d, dist0, dist1))
worst = np.array(worst)
print("min margin overall:", worst[:, 0].min())
for lo, hi in [(0, 0.02), (0.02, 0.1), (0.1, 0.3), (0.3, 1e9)]:
    sel = (np.minimum(worst[:, 3], worst[:, 4]) >= lo) & (np.minimum(worst[:, 3], worst[:, 4]) < hi)
    k = np.argmin(worst[sel, 0]); w = worst[sel][k]
    print(f"  distance to nearest equality point in [{lo},{hi}): min margin {w[0]:.3e} at beta={w[1]:.4f}, delta={w[2]:.4f}")
# local exponents near the two equality points
for eps in [1e-2, 1e-3, 1e-4]:
    print(f"near identity, beta=0, delta={eps}: margin {rhs(0, eps) - lhs(0, eps):.3e} (pred 7/18 delta^2 = {7/18*eps**2:.3e});  "
          f"near atom, beta=sqrt2-eps, delta=1-eps: margin {rhs(np.sqrt(2)-eps, 1-eps) - lhs(np.sqrt(2)-eps, 1-eps):.3e}")
