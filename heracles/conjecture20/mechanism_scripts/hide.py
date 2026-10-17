# Residual of hiding feature l's sign-flipped character from the binding set B using the other features' outputs
# (riding modes x_j, free) and conjugate partners y_j (incl. own).  R_l = min sum_{delta in B} mu |c_l + sum x c + y conj c| / K_l(mu).
import numpy as np, cvxpy as cp, sys
from prog import solve, kappa
s=float(sys.argv[1])
for p in [int(a) for a in sys.argv[2:]]:
    G,y,cov=solve(p,s); F=(p-1)//2; S=np.nonzero(y)[0]+1
    Bh=np.nonzero(cov<G*(1+1e-6))[0]+1; B=np.concatenate([Bh,p-Bh])   # symmetric binding set
    rho=np.exp(2j*np.pi/p)
    c=lambda j,d:1-rho**(-j*d)
    # dual weights: uniform on B (simple choice) and also LP dual approx via equal weights
    mu=np.ones(len(B))/len(B)
    out=[]
    for l in S:
        others=[j for j in S if j!=l]
        x=cp.Variable(len(others),complex=True); yv=cp.Variable(len(S),complex=True)
        Cl=np.array([c(l,d) for d in B])
        Mx=np.array([[c(j,d) for j in others] for d in B]) if others else np.zeros((len(B),0))
        My=np.array([[np.conj(c(j,d)) for j in S] for d in B])
        expr=Cl+(Mx@x if others else 0)+My@yv
        prob=cp.Problem(cp.Minimize(cp.sum(cp.multiply(mu,cp.abs(expr)))))
        prob.solve()
        Kl=np.sum(mu*(1-np.cos(2*np.pi*l*B/p)))
        out.append(prob.value/Kl)
    print(f"p={p} s={s} K*={len(S)} |B|={len(B)}  R_l (riding+partners free) = {np.round(out,3)}  min={min(out):.3f}",flush=True)
