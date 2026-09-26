# 3a.7 (subset monotonicity) and 3a.8(a) (Riesz density on supports whose 3^K signed sums of 2k are distinct).
import itertools, numpy as np
rng = np.random.default_rng(5)

def signed_sums(p, S):
    return [sum(e * 2 * k for e, k in zip(eps, S)) % p for eps in itertools.product((-1, 0, 1), repeat=len(S))]

def distinct(p, S):
    s = signed_sums(p, S); return len(set(s)) == len(s)

def zero_free(p, S):   # plain dissociate: no nontrivial {-1,0,1}-relation
    return sum(1 for v in signed_sums(p, S) if v == 0) == 1

def riesz(p, S, lam):
    x = np.arange(p); R = np.ones(p)
    for k, l in zip(S, lam):
        R *= 1 + 2 * np.real(l * np.exp(-2j * np.pi * 2 * k * x / p))   # 1 + 2|l| cos(...) >= 0
    return R / p

bad_distinct = bad_zero_free = n_distinct = n_zf_only = 0
for p in (11, 13, 17, 19, 23, 29, 31, 37):
    h = (p - 1) // 2
    for K in (2, 3, 4):
        for S in itertools.combinations(range(1, h + 1), K):
            if not zero_free(p, S):
                continue
            lam = 0.5 * np.exp(2j * np.pi * rng.uniform(size=K)) * rng.uniform(0.5, 1, K)
            w = riesz(p, S, lam)
            mom = [np.sum(w * np.exp(2j * np.pi * 2 * k * np.arange(p) / p)) for k in S]   # E_w[w^{+2kx}] pairs with lam
            err = max(abs(w.sum() - 1), max(abs(m - l) for m, l in zip(mom, lam)))
            ok = err < 1e-12 and w.min() >= -1e-15
            if distinct(p, S):
                n_distinct += 1; bad_distinct += (not ok)
            else:
                n_zf_only += 1; bad_zero_free += (not ok)
print(f"distinct-sums supports: {n_distinct}, Riesz fails on {bad_distinct}")
print(f"zero-free-but-not-distinct supports: {n_zf_only}, Riesz moments wrong on {bad_zero_free}")
# 3a.7: depth never drops when frequencies are removed (min over a sub-family of directions)
x13 = np.arange(13)
def depth_est(p, S, trials=20000):
    ze = np.exp(-2j * np.pi * 2 * np.outer(S, np.arange(p)) / p); best = 9
    for _ in range(trials):
        z = rng.normal(size=len(S)) + 1j * rng.normal(size=len(S))
        best = min(best, np.real(z @ ze).max() / np.abs(z).sum())
    return best
print("3a.7 sample (p=13): depth{1,3,4} ~", round(depth_est(13, [1, 3, 4]), 3), "<= depth{1,3} ~", round(depth_est(13, [1, 3]), 3), "<= depth{1} =", round(np.cos(np.pi / 13), 3))
