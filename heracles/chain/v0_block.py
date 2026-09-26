# V0: the block attains gamma_sym (Prop 7.3) and the programme solver reproduces astra's certified values.
import numpy as np
from common import *
astra = {3: 1.5, 5: 0.8838834764831844, 7: 0.6735753140545634, 11: 0.5357259257995056,
         13: 0.4977720672100500, 17: 0.4483405600853964}
for p in (3, 5, 7, 11, 13, 17):
    Gam, y = programme(p, 1.5)
    net = block_net(p, y)
    z = table(p, net)
    a = np.arange(p)
    target = sum(y[k - 1] * np.cos(2 * np.pi * k * (a[:, None, None] + a[None, :, None] - a[None, None, :]) / p)
                 for k in range(1, len(y) + 1) if y[k - 1] > 0)
    G, J = spectrum(z)
    junk = max(np.abs(J[T]).max() for T in J) / np.abs(G).max()
    g = gamma(p, net)
    print(f"p={p:2d} Gamma={Gam:.13f} astra_diff={Gam - astra[p]:+.1e} K={int((y > 0).sum())} "
          f"table_err={np.abs(z - target).max():.1e} junk/clean={junk:.1e} "
          f"margin-Gamma={margin(z) - Gam:+.1e} gamma/gamma_sym={g / gamma_sym(p, Gam):.12f}")
