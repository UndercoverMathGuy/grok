# gamma_max(n) = (3/4) sup |tau|/sigma^3, tau = sum_{T,T' disjoint nonempty, ordered} a_T a_T' conj(a_{T u T'}),
# sigma^2 = (1/2) sum_T |a_T|^2.  Nonnegative real a suffices for the sup (|tau| <= sum |a||a||a|).
import numpy as np, itertools
from scipy.optimize import minimize
for n in [2,3,4,5]:
    sets=[m for m in range(1,2**n)]; idx={m:i for i,m in enumerate(sets)}
    trip=[(idx[a],idx[b],idx[a|b]) for a in sets for b in sets if a&b==0]
    I=np.array(trip)
    def neg(x):
        x=np.abs(x); x=x/np.linalg.norm(x)          # sum x^2 = 1 -> sigma^3 = (1/2)^{3/2}
        return -np.sum(x[I[:,0]]*x[I[:,1]]*x[I[:,2]])
    best=0; rng=np.random.default_rng(0)
    for t in range(200):
        r=minimize(neg,rng.random(len(sets)),method='L-BFGS-B'); best=max(best,-r.fun)
    g=0.75*best/(0.5**1.5); f=(np.sqrt(g*g+4)-g)/2
    print(f"n={n}: sup tau-form={best:.5f}  gamma_max={g:.4f}  f(gamma_max)={f:.4f}",flush=True)
