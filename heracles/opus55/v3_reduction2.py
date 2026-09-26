# Two-variable reduction of (V3):  with |G|=1, max_T|J_T| <= delta, e1 = sum|c| >= 1,
#   e2 >= max(0,(1 - delta*e1)/3),  e3 >= max(0,(1 - 3 delta e1 - delta e1^2 - delta^2 e1)/27)
#   cost >= F(delta,e1) := C_r * int_0^inf log(1 + e1 t + e2 t^2 + e3 t^3) t^{-1-r} dt
# Claim needed:  min_{e1>=1} F(delta,e1) >= 3^{1-r} (1 - delta/2)^r   for all delta in [0,2].
import numpy as np
from scipy.integrate import quad
r=2/3; Cr=r*np.sin(np.pi*r)/np.pi
def F(d,e1):
    mu=min(e1,d+2*np.sqrt(2*e1*max(e1-1,0)))          # mirror bound |U_T| <= delta + 2 sqrt(2 e1 (e1-1))
    e2=max(0,(3-3*d*mu)/9); e3=max(0,(1-3*d*mu-d*mu**2-d*d*mu)/27)
    f=lambda u: np.log1p(e1*np.exp(u)+e2*np.exp(2*u)+e3*np.exp(3*u))*np.exp(-r*u)   # t=e^u
    return Cr*quad(f,-60,60,limit=400)[0]
print("sanity F(0,1) (block) =",F(0,1)," vs 3^{1-r} =",3**(1-r))
worst=np.inf
for d in np.r_[np.linspace(0,0.2,41),np.linspace(0.2,2,37)]:
    e1s=np.r_[1+np.logspace(-8,0,120),np.linspace(2,3,41),np.linspace(3,50,48)]
    Fm=min(F(d,e) for e in e1s); target=3**(1-r)*max(0,1-d/2)**r
    ratio=Fm/target if target>0 else np.inf
    worst=min(worst,ratio)
    if d in (0,0.01,0.05,0.1,0.2) or ratio<1: print(f"delta={d:.3f}: min_e1 F / target = {ratio:.5f}",flush=True)
print("worst ratio over grid:",worst)
