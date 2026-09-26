# 3a.1-3a.4: worst <= average; pointwise Fourier formula for the gap; three bins (X = 0 for single-frequency nets);
# D/-D averaging -> real coverage weights and clean <= |G_k|.
import numpy as np
from common import *
rng = np.random.default_rng(3)
res = dict(avg=-9, pure_formula=0, impure_formula_err=0, impure_X_rebuilt=0, pm_avg=0, clean_le=-9)
for p in (5, 7, 11, 13):
    w = np.exp(2j * np.pi / p); h = (p - 1) // 2
    for pure in (True, False):
        for t in range(10):
            net = random_net(p, 9, rng, pure=pure)
            z = table(p, net); g = gaps(z); m = g[:, :, 1:].min()
            # 3a.1 worst <= any weighted average (random weighting over inputs)
            wt = rng.dirichlet(np.ones(p * p)).reshape(p, p)
            for D in range(1, p):
                res['avg'] = max(res['avg'], m - (wt * g[:, :, D]).sum())
            # 3a.2/3a.3: gap = (2/p^3) sum_k Re[(1 - w^{kD}) (G_k + J1 w^{-2ka} + J2 w^{-2kb} + J12 w^{-2k(a+b)})] + X
            G, J = spectrum(z)
            a = np.arange(p)[:, None]; b = np.arange(p)[None, :]
            formula = np.zeros((p, p, p))
            for D in range(1, p):
                tot = 0
                for k in range(1, h + 1):
                    inner = G[k] + J['1'][k] * w ** (-2 * k * a) + J['2'][k] * w ** (-2 * k * b) + J['12'][k] * w ** (-2 * k * (a + b))
                    tot = tot + np.real((1 - w ** (k * D)) * inner)
                formula[:, :, D] = 2 / p ** 3 * tot
            err = np.abs(formula - g)[:, :, 1:].max() / np.abs(g).max()
            if pure:
                res['pure_formula'] = max(res['pure_formula'], err)
            else:
                res['impure_formula_err'] = max(res['impure_formula_err'], err)
                # X rebuilt from all Fourier components that are not clean / same-|k| junk must close the gap exactly
                Zh = np.fft.fftn(z); X = np.zeros((p, p, p))
                for al in range(p):
                    for be in range(p):
                        for ga in range(p):
                            if ga == 0:
                                continue
                            ka, kb, kg = [min(x, p - x) for x in (al, be, ga)]
                            if ka == kb == kg:      # clean or same-|k| junk: already in the formula
                                continue
                            X += np.real(Zh[al, be, ga] * np.exp(2j * np.pi * (al * a[:, :, None] + be * b[:, :, None] + ga * ((a + b)[:, :, None])) / p)
                                         * (1 - np.exp(-2j * np.pi * ga * np.arange(p)[None, None, :] / p))) / p ** 3
                res['impure_X_rebuilt'] = max(res['impure_X_rebuilt'], np.abs(formula + X - g)[:, :, 1:].max() / np.abs(g).max())
            # 3a.4: average of gap at D and -D, over any row weighting, has real coverage weights; clean part <= sum |G_k| cov
            for D in range(1, p):
                pm = 0.5 * (g[:, :, D] + g[:, :, p - D])
                cov = 1 - np.cos(2 * np.pi * np.arange(1, h + 1) * D / p)
                pred = 2 / p ** 3 * sum(cov[k - 1] * np.real(G[k] + J['1'][k] * w ** (-2 * k * a) + J['2'][k] * w ** (-2 * k * b)
                                                             + J['12'][k] * w ** (-2 * k * (a + b))) for k in range(1, h + 1))
                if pure:
                    res['pm_avg'] = max(res['pm_avg'], np.abs(pm - pred).max() / np.abs(g).max())
                res['clean_le'] = max(res['clean_le'], pm.mean() - 2 / p ** 3 * (cov * np.abs(G[1:])).sum())
print({k: float(f"{v:.2e}") for k, v in res.items()})
print("need: avg<=0, pure_formula~0, impure_formula_err>0 (X present), impure_X_rebuilt~0, pm_avg~0, clean_le<=0")
