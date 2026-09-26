# Adversarial check of the NEW pair-offset floor for general (3-pattern) junk:
#   (dip_D + dip_{-D})/2 >= (2/3) (sum_k cov_k(D) eta_k^2)^{1/2}      [claimed PROVED]
# Minimise the ratio over J by Adam (smoothed max) from structured + random starts; report min ratio (must be >= 2/3).
import numpy as np, torch
from mo import Body, cov, to_J, structured_starts
rng = np.random.default_rng(11)
worst = (9, None)
for p in (5, 7, 11, 13, 17, 23, 31):
    h = (p - 1) // 2
    for trial in range(6):
        K = int(rng.integers(1, min(h, 6) + 1)); T = sorted(rng.choice(np.arange(1, h + 1), K, replace=False).tolist())
        if trial == 0: T = list(range(1, h + 1)); K = h
        D = int(rng.integers(1, p)); c = torch.tensor(cov(p, np.array(T), np.array([D]))[:, 0])
        body = Body(p, T, [D, p - D], [0.5, 0.5])
        def ratio(J, beta=None):
            return (body.lamt * body.dips(J, beta)).sum() / torch.sqrt((c * (torch.abs(J) ** 2).sum(0)).sum())
        starts = structured_starts(p, T, '3', rng) + [rng.normal(size=6 * K) for _ in range(6)]
        for s in starts:
            v = torch.tensor(np.array(s, float), requires_grad=True); opt = torch.optim.Adam([v], lr=0.03)
            for beta in (20., 80., 400.):
                for _ in range(200):
                    opt.zero_grad(); J = to_J(v, K, '3'); loss = ratio(J / (torch.abs(J).max() + 1e-12), beta * K); loss.backward(); opt.step()
            with torch.no_grad():
                r = float(ratio(to_J(v, K, '3')))
            if r < worst[0]:
                worst = (r, (p, T, D))
    print(f"after p={p}: min ratio so far {worst[0]:.4f} at {worst[1]}  (claim >= 0.6667; Parseval-per-pattern bound would be 0.5774)", flush=True)
