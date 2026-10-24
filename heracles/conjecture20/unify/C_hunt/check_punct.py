# check Prop (punctured IF block): n equal aligned atoms, xi_kt = zeta^k -> a_T = A [n | |T|], G = (nu/n)^{s-1};
# also the realised network (class-C model, single feature) has positive or non-positive margin? (report it)
import itertools, numpy as np, hlib as H
for n in range(2, 7):
    nu = n + 1; z = np.exp(2j * np.pi / n); A = 1.0
    b = np.full(n, A / n); xi = np.array([[z ** k] * n for k in range(n)])
    aT = {T: np.sum(b * np.prod(xi[:, list(T)], axis=1)) for k in range(1, n + 1) for T in itertools.combinations(range(n), k)}
    nz = {T: abs(v) for T, v in aT.items() if abs(v) > 1e-12}
    for s in [1.2, nu / 2]:
        r = 1 / s; G = A / (nu ** (1 - s) * (n * (A / n) ** r) ** s)
        print(f"n={n} s={s:.3f}: nonzero a_T={ {T: round(v, 12) for T, v in nz.items()} } G={G:.10f} (nu/n)^(s-1)={(nu / n) ** (s - 1):.10f} theta={(1 - 1 / G):.6f}")
    for p in [5, 7, 11]:
        if p ** n > 2e5: continue
        cc = H.CC(p, n, [1] * n, nu / 2)
        x = cc.pack(b.astype(complex), np.angle(xi))
        print(f"    realised single feature p={p}: margin ratio to IF price = {cc.gnorm(x) / (nu ** (1 - nu / 2) * (1 - np.cos(2 * np.pi / p))):.4f}")
