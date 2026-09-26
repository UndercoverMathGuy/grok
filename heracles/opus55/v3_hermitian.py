# Hermitian (mirror-free) reduction of (V3), r=2/3.  Normalise G=1, delta = max_T |J_T|, e1 = sum|c| >= 1.
#   M_h = sum |c_l| m_l m_l^*  (PSD, diag e1),  sum (3|c|)^r >= sum lam^r   [Rotfel'd]
#   lam = e1 + mu, sum mu = 0, sum mu^2 = 2 sum|K_T|^2 <= 6 kappa^2, kappa = delta + sqrt(2 e1 (e1-1)), lam >= 0
# Need: H(delta,e1) := min sum lam^r  >=  3 (1 - delta/2)^r   for all delta in [0,2), e1 >= 1.
import numpy as np
r=2/3
def Hmin(d,e1,n=721):
    k=d+np.sqrt(2*e1*max(e1-1,0)); rho=np.sqrt(6)*k           # ||mu||_2 <= rho
    # parametrise the circle {sum mu=0, ||mu||=t} for t<=rho; concave min is on the outer boundary or the lam>=0 faces
    a=np.array([1,-1,0])/np.sqrt(2); b=np.array([1,1,-2])/np.sqrt(6)
    th=np.linspace(0,2*np.pi,n)
    best=np.inf
    for t in np.linspace(0,rho,41):
        mu=t*(np.outer(np.cos(th),a)+np.outer(np.sin(th),b)); lam=e1+mu
        ok=(lam>=-1e-12).all(1)
        if ok.any(): best=min(best,(np.clip(lam[ok],0,None)**r).sum(1).min())
    return best
worst=np.inf; worst_at=None
for d in np.r_[np.linspace(0,0.5,51),np.linspace(0.5,1.99,31)]:
    target=3*(1-d/2)**r
    m=min(Hmin(d,e1) for e1 in np.r_[1+np.logspace(-9,-1,40),np.linspace(1.1,4,59)])
    if m/target<worst: worst=m/target; worst_at=d
    if d in (0,0.1,0.2,0.25,0.3,0.35,0.5,1.0): print(f"delta={d:.2f}: min_e1 H / target = {m/target:.5f}",flush=True)
print(f"worst ratio {worst:.5f} at delta={worst_at:.3f}")
