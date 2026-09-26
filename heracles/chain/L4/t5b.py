import sys; sys.path.insert(0, '..')
import numpy as np
from transfer_flip import depth
w = 0
for As in np.linspace(0.05, 0.9, 10):
    for Ao in np.linspace(0, As, 6):
        c, _ = depth(As, Ao, n=40); w = max(w, c / As)
print("max c/A_s:", round(w, 4))
