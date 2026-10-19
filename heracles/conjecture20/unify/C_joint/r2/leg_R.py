import sys, numpy as np
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/C_joint')
from cjlib import gamma_lp
for (p, n, s) in [(347, 2, 1.5), (383, 2, 1.5), (443, 3, 2.0), (467, 3, 2.0)]:
    G, y = gamma_lp(p, s, restarts=8)
    K = (p-1)//2; e = (p/2)/(G*K**s); tau = (1+1/n)**(s-1)
    print(f"p={p} n={n} s={s}: Gamma_s >= {G:.5f} (support {np.count_nonzero(y)}), uniform efficiency e <= {e:.4f}, "
          f"gamma/gamma_IF <= tau (1-2/sqrt p) e = {tau*(1-2/np.sqrt(p))*e:.4f}", flush=True)
