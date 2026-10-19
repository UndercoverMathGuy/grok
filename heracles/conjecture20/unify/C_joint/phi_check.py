# One-variable inequality for the single-T payment (aligned atoms):
#   f(d) = nu^{1-s} Phi_n(d)^s >= 1 - c_n d on [0,1],  Phi_n(d) = 1/2[(1+d)^r (nu+(n-1)d)^{1-r} + (1-d)^r (nu-(n-1)d)^{1-r}]
# and test concavity of f and of log-forms.
import numpy as np
def Phi(n, r, d):
    nu = n+1
    return 0.5*((1+d)**r*(nu+(n-1)*d)**(1-r) + (1-d)**r*(nu-(n-1)*d)**(1-r))
worst = (1e9, None); conc = (1e9, None)
d = np.linspace(0, 1, 20001)
for n in range(1, 61):
    nu = n+1
    for s in np.linspace(1.0001, nu/2 if nu/2 > 1.0001 else 1.0001, 60):
        r = 1/s; c = 1-(n/nu)**(s-1)
        f = nu**(1-s)*Phi(n, r, d)**s
        gap = f - (1-c*d)
        # relative slack in the interior
        m = gap[1:-1].min()
        if m < worst[0]: worst = (m, (n, s, d[1:-1][gap[1:-1].argmin()]))
        sec = f[2:] - 2*f[1:-1] + f[:-2]
        if sec.max() > -conc[0] if False else False: pass
        cm = sec.max()
        if -cm < conc[0]: conc = (-cm, (n, s, d[1:-1][sec.argmax()]))
print("min interior slack f-(1-c d):", worst)
print("max second difference of f (concavity needs <=0):", -conc[0], conc[1])
