# Global regime: how much of a maximally concentrated feature (one class-C neuron, a Riesz product) can averaged
# (second-moment / oscillation) detection see at the binding offset delta_0 = +-1 (argmin kappa_1)?
#   det = (1/2)(E F_d^2/osc F_d + E F_-d^2/osc F_-d) / (|beta| kappa_1(1)),   depth = (1/2)(m_d + m_-d)/(|beta| kappa_1(1))
# Payment a single atom can claim at s: 1 - nu^{1-s}.  (W9: one neuron has h <= 0, i.e. depth >= 1.)
import numpy as np, math
from sclib import *
rng = np.random.default_rng(2)
for (p, n) in [(5, 2), (7, 2), (11, 2), (13, 2), (5, 3), (7, 3), (5, 4), (5, 5), (5, 6), (7, 4)]:
    nu = n + 1; s = nu / 2
    dets = []; deps = []
    for trial in range(20):
        phis = rng.uniform(0, 2 * np.pi, n)
        net = classC_net(p, n, [(1, np.ones(nu), phis)], 2.0)
        g = net.gaps(); beta = abs(net.clean()[0]); k1 = 1 - math.cos(2 * math.pi / p)
        F = g - g.mean(0)
        det = 0.5 * sum((F[:, i] ** 2).mean() / np.ptp(F[:, i]) for i in [0, p - 2]) / (beta * k1)
        dep = 0.5 * sum(-F[:, i].min() for i in [0, p - 2]) / (beta * k1)
        dets.append(det); deps.append(dep)
    print('(p,n)=(%2d,%d): averaged detection %.3f-%.3f ; true depth %.3f-%.3f (x|beta|kappa_min); payment at b=2: 1-nu^{1-s} = %.3f'
          % (p, n, min(dets), max(dets), min(deps), max(deps), 1 - nu ** (1 - s)))
