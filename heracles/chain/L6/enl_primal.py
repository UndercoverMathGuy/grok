# Flip-enlarged l2-polyball depth by the primal LP: t_max(u) = max t with t*u in hull{ (s1 chi(a), s2 chi(b), s1 s2 chi(a+b)) },
# u = (u_k) per-frequency unit vectors in C^3; depth = min over u of t_max(u) (search = UPPER estimate).
# Starts include the flattest single-pattern direction (value A_s) so we see whether mixing patterns goes lower.
import sys, json, numpy as np
from scipy.optimize import linprog, minimize
from tools import depth_polar
p = int(sys.argv[1]); T = [int(x) for x in sys.argv[2].split(',')]; K = len(T); rng = np.random.default_rng(4)
x = np.arange(p); chi = np.exp(-2j * np.pi * 2 * np.outer(T, x) / p)          # [K, p]
A_, B_ = np.meshgrid(x, x, indexing='ij'); A_ = A_.ravel(); B_ = B_.ravel(); S_ = (A_ + B_) % p
cols = []
for s1, s2 in ((1, 1), (-1, 1), (1, -1)):
    V = np.vstack([s1 * chi[:, A_], s2 * chi[:, B_], s1 * s2 * chi[:, S_]])     # [3K, p^2]
    cols.append(V)
V = np.hstack(cols); n = V.shape[1]
Aeq_base = np.vstack([V.real, V.imag, np.ones(n)])
def tmax(u):
    u = u.reshape(3, K); u = u / np.linalg.norm(u, axis=0, keepdims=True)
    uu = u.ravel()
    r = linprog(np.r_[np.zeros(n), -1], A_eq=np.column_stack([Aeq_base, -np.r_[uu.real, uu.imag, 0.0]]),
                b_eq=np.r_[np.zeros(6 * K), 1], bounds=[(0, None)] * n + [(None, None)], method='highs')
    return -r.fun if r.status == 0 else 0.0
def f(v):
    return tmax(v[:3 * K] + 1j * v[3 * K:])
As, th = depth_polar(p, T, True, rng, starts=10, iters=200)
single = np.zeros((3, K), complex); single[0] = np.exp(1j * th)           # target phases of the flattest two-sided direction
starts = [np.r_[(single + 1e-3 * rng.normal(size=(3, K))).real.ravel(), (single + 1e-3 * rng.normal(size=(3, K))).imag.ravel()]]
starts += [rng.normal(size=6 * K) for _ in range(int(sys.argv[3]) if len(sys.argv) > 3 else 3)]
best = 9
for s in starts:
    r = minimize(f, s, method='Powell', options=dict(maxfev=2500, xtol=1e-5, ftol=1e-8))
    best = min(best, r.fun, f(s))
    print(json.dumps(dict(start_value=round(f(s), 4), end_value=round(r.fun, 4), best=round(best, 4))), flush=True)
print(json.dumps(dict(p=p, T=T, A_s=round(As, 4), enlarged_depth_upper=round(best, 4))), flush=True)
