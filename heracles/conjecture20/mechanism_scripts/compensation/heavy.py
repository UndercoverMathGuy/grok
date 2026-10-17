import numpy as np,sys
from fb import gstar
s=float(sys.argv[1]); c=0.5*1.5**(s-1)
for rho in [float(x) for x in sys.argv[2:]]:
    rng=np.random.default_rng(int(rho*1000))
    g,N,v=gstar(s,(rho,rho,rho),rng,Ns=(2,3,4,5,6),trials=60)
    G2=3*rho-rho**3/c**2-1 if rho<=c else 2*c-1
    print(f"s={s} rho={rho}: heavy G*={g:.6f} (N={N})  two-atom G2={G2:.6f}",flush=True)
