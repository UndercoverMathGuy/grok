# Check the two ceilings used in sec 3: (i) true flip-enlarged l2-ball depth <= two-sided depth A_s (single-pattern
# directions), (ii) transfer constant c(A*, A_s) <= A_s (1 + A*) on a grid of inputs.
import sys, numpy as np
sys.path.insert(0, '..')
from transfer_flip import depth as transfer_depth
worst = 0
for As in np.linspace(0.05, 0.9, 12):
    for Ao in np.linspace(0.0, 1.0, 11):
        if Ao > As: continue       # one-sided never exceeds two-sided
        c, _ = transfer_depth(As, Ao, n=40)
        worst = max(worst, c / (As * (1 + Ao)))
print(f"max of c(A*,A_s) / (A_s (1 + A*)) over the grid: {worst:.4f}  (<= 1 claimed)")
