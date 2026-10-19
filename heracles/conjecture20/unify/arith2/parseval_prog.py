# Maximal kappa-mass of a dilated uniform progression d*{1..q}:  M_q = max_x sum_{j=1}^q (1 - cos j x)  (x real; every
# offset of every dilation at every prime is such an x).  The relations case with cheap set = d*{1..q} is closed by
# Parseval at EVERY offset iff c^2 M_q <= 1, c = 1-(2/3)^{s-1}.  Rigorous: dense grid + |d/dx| <= sum j bound.
import numpy as np
for s in [1.5, 4/3, 1.25, 1.2]:
    c = 1-(2/3)**(s-1); cap = 1/c**2; qmax = None
    for q in range(1, 200):
        x = np.linspace(0, np.pi, 200001); h = x[1]-x[0]
        P = (1-np.cos(np.outer(x, np.arange(1, q+1)))).sum(1)
        Mq_upper = P.max() + q*(q+1)/2*h/2      # Lipschitz slack (derivative bound sum_j j)
        if Mq_upper <= cap: qmax = q
        else: break
    print(f"s={s:.4f} (b={3/s:.3f}): capacity 1/c^2={cap:.2f};  largest q with max_x kappa-mass <= capacity: q={qmax}"
          f"   (M_q/q -> 1.2172 as q grows)", flush=True)
