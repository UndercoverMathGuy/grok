# 3a.5: moments of a weighting at one character = regular p-gon; worst-direction depth = cos(pi/p).
import numpy as np
from scipy.optimize import linprog
for p in (3, 5, 7, 11, 13):
    k = 1; x = np.arange(p); chi = np.exp(-2j * np.pi * 2 * k * x / p)
    worst = 9
    for phi in np.linspace(0, 2 * np.pi, 721):
        # maximise t s.t. sum w chi = t e^{i phi}, w >= 0, sum w = 1   (vars: w_0..w_{p-1}, t)
        u = np.exp(1j * phi)
        Aeq = np.vstack([np.r_[chi.real, -u.real], np.r_[chi.imag, -u.imag], np.r_[np.ones(p), 0]])
        r = linprog(np.r_[np.zeros(p), -1], A_eq=Aeq, b_eq=[0, 0, 1], bounds=[(0, None)] * p + [(None, None)])
        worst = min(worst, -r.fun)
    print(f"p={p:2d}  min over directions of reachable depth = {worst:.6f}   cos(pi/p) = {np.cos(np.pi / p):.6f}")
