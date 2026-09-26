# D* = min over atom pairs (mu_a, mu_b), mu=(1,v,w,vw), of  min_{alpha,beta in C} || e0 - alpha mu_a - beta mu_b ||_inf
# Then every pair of atoms leaves residual mass rho >= D* - delta (each coordinate of the rest is <= its mass).
# WLOG mu_a = (1,1,1,1) (rotate v,w); scan mu_b over the torus.
import numpy as np
from scipy.optimize import minimize
e0=np.array([1,0,0,0],complex)
def dist_inf(vb,wb):
    ma=np.ones(4,complex); mb=np.array([1,vb,wb,vb*wb])
    best=np.inf
    for start in [np.zeros(4),np.array([1/3,0,1/3,0]),np.random.default_rng(0).normal(0,.3,4)]:
        f=lambda z: np.abs(e0-(z[0]+1j*z[1])*ma-(z[2]+1j*z[3])*mb).max()
        res=minimize(f,start,method='Nelder-Mead',options={'xatol':1e-10,'fatol':1e-12,'maxiter':4000})
        best=min(best,res.fun)
    return best
g=np.exp(2j*np.pi*np.arange(72)/72)
vals=np.array([[dist_inf(a,b) for b in g] for a in g])
i,j=np.unravel_index(vals.argmin(),vals.shape)
print(f"D* (grid 72x72) = {vals.min():.6f} at v_b angle {np.angle(g[i])/np.pi*180:.1f} deg, w_b angle {np.angle(g[j])/np.pi*180:.1f} deg")
om=np.exp(2j*np.pi/3); print("block pair (omega,omega):",dist_inf(om,om))
