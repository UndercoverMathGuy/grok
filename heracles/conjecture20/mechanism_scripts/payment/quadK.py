# Global quadratic constant obtainable from the proof chain:
#  bonus <= L [ min_{C>=0} (C d + sqrt(1+C^2-2Cy)) - S(b1,b2,y) ],  S = ((1/3) sum nu^r)^s with worst phase
#  ratio K = bonus*L/(|B|^2+|C|^2+|D|^2) <= [..]/(b1^2+b2^2+d^2)   (and |A|<=L)
import numpy as np, sys
from scipy.optimize import minimize
s=float(sys.argv[1]); r=1/s
def S(b1,b2,y):
    P=b1*b1+b2*b2+y*y; q=max(-2*b1*b2*y,P-1)
    nu=np.maximum(1+np.roots([1,0,-P,-q]).real,0)
    return np.mean(nu**r)**s
def mC(d,y):
    # min_{C>=0} C d + sqrt(1+C^2-2Cy)
    if d>=y: return 1.0
    return d*y+np.sqrt((1-d*d)*(1-y*y))
def ratio(v):
    b1,b2,y,d=np.clip(v,1e-9,1)
    num=mC(d,y)-S(b1,b2,y)
    return num/(b1*b1+b2*b2+d*d)
rng=np.random.default_rng(0); best=0
for t in range(3000):
    v0=rng.uniform(0,1,4)*rng.choice([0.05,0.3,1.0])
    res=minimize(lambda v:-ratio(v),v0,method='Nelder-Mead',options={'xatol':1e-9,'fatol':1e-12,'maxiter':3000})
    if -res.fun>best: best=-res.fun; bv=np.clip(res.x,0,1)
print(f"s={s}: sup K from chain = {best:.4f} at b1,b2,y,d = {np.round(bv,4)}  (numerical true sup 0.275 at s=1.5; local sharp (1-r)/(8+r)*3={(1-r)/(8+r)*3:.4f})")
