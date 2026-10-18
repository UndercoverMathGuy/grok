# Referee: dense-grid sanity check of the doubling-lemma residual R(x) = 2 eps P_q(2x) + lam (1-cos x) - E_q(x) >= 0,
# eps = 1.01/q, lam = 1.2 q, for EVERY q in [84, 600]; E_q computed by direct summation (not the closed form);
# also the identity P_2q(x) = 2 P_q(2x) + E_q(x).
import numpy as np
x = np.concatenate([np.linspace(1e-5, np.pi, 20001), np.geomspace(1e-6, 1e-2, 2001)])
worst = np.inf; idw = 0
for q in list(range(84, 161)) + list(range(161, 601, 13)):
    j = np.arange(1, q+1)
    E = np.zeros_like(x); P2 = np.zeros_like(x); Pq2 = np.zeros_like(x); P2q = np.zeros_like(x)
    for j0 in range(0, q, 100):
        jj = j[j0:j0+100]
        E += (np.cos(2*np.outer(x, jj))-np.cos(np.outer(x, 2*jj-1))).sum(1)
        Pq2 += (1-np.cos(np.outer(2*x, jj))).sum(1)
    for j0 in range(1, 2*q+1, 200):
        jj = np.arange(j0, min(2*q, j0+199)+1)
        P2q += (1-np.cos(np.outer(x, jj))).sum(1)
    idw = max(idw, np.abs(P2q-2*Pq2-E).max())
    R = 2*(1.01/q)*Pq2+1.2*q*(1-np.cos(x))-E
    rel = (R/np.maximum(x*x, 1e-300)).min()   # R ~ x^2 near 0
    worst = min(worst, rel)
print(f"q in 84..160 (all) and 161..600 (step 13): max identity error {idw:.1e};  min_x R(x)/x^2 over all q = {worst:.4f} (> 0 needed)")
