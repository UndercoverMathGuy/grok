# symmetric 3-atom ansatz for n=4: atoms d (xi=1), c e^{+-i psi} (xi=e^{+-i theta} in every slot); beta_k = d + 2c cos(k theta + psi), A = beta_0
import numpy as np, itertools
from scipy.optimize import minimize
from fams import collision_free_families
n=4; Ts,V,F=collision_free_families(n)
sizes=[len(Ts[i]) for i in range(len(Ts))]
sig=sorted(set(tuple(sorted(sizes[i] for i in f)) for f in F))
print("collision-free family size-signatures (n=4):",sig)
def ratio(x,s,detail=False):
    th,ps,u=x  # d = 1 - 2c cos psi with c = u
    c=u; d=1-2*c*np.cos(ps)
    beta=lambda k:d+2*c*np.cos(k*th+ps)
    Q=abs(d)**(1/s)+2*c**(1/s); bonus=1-5**(1-s)*Q**s
    den=max(sum(abs(beta(k)) for k in sg) for sg in sig)
    if detail: return bonus,den,[beta(k) for k in range(5)],d,c
    return bonus/den
for s in (2.0,2.25,2.5):
    best=(-9,None)
    for t in range(400):
        x0=np.array([np.random.uniform(0,np.pi),np.random.uniform(-np.pi,np.pi),np.random.uniform(0.05,0.8)])
        r=minimize(lambda x:-ratio(x,s),x0,method='Nelder-Mead',options={'xatol':1e-12,'fatol':1e-14,'maxiter':5000})
        if -r.fun>best[0]: best=(-r.fun,r.x)
    b,den,bet,d,c=ratio(best[1],s,True)
    print(f"s={s}: ratio {best[0]:.5f}  theta={best[1][0]:.6f} psi={best[1][1]:.6f} c={c:.6f} d={d:.6f} bonus={b:.5f} den={den:.5f} beta_k={np.round(bet,5)}")
