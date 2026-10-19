import sys; sys.path.insert(0,'.')
import numpy as onp
from mfnet import MF, seed, optimise
rng=onp.random.default_rng(0)
for p,n in [(5,3),(5,4),(7,3)]:
    mf=MF(p,n); nu=n+1
    z=seed(mf,nu,'if',rng,0.0)
    print(p,n,"IF block ratio",mf.ratio(z,nu),"clean",mf.clean(z,nu), "Gamma",mf.G)
