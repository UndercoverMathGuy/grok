import numpy as np, itertools, mpmath as mp
from scipy.optimize import minimize, brentq
from fams import collision_free_families
import sympy
mp.mp.dps=50
# --- verify collision-free family lists with exact integer arithmetic
def cf_exact(n):
    Ts=[T for k in range(1,n+1) for T in itertools.combinations(range(n),k)]
    V=[tuple(1 if t in T else 0 for t in range(n)) for T in Ts]
    targets=set(V)|set(tuple(-x for x in v) for v in V)
    out=[]
    for k in range(1,n+1):
        for F in itertools.combinations(range(len(Ts)),k):
            M=sympy.Matrix([V[i] for i in F])
            if M.rank()<k: continue
            ok=True
            for sig in itertools.product((-1,0,1),repeat=k):
                if sum(1 for x in sig if x)<2: continue
                vec=tuple(sum(sig[j]*V[F[j]][t] for j in range(k)) for t in range(n))
                if vec in targets: ok=False; break
            if ok: out.append(F)
    return Ts,out
def check(n,s,atoms):  # atoms: list of (beta (mp complex), list of slot phases (mp))
    Ts,F=cf_exact(n); nu=n+1
    A=sum(b for b,_ in atoms)
    BT=[sum(b*mp.e**(1j*sum(ph[t] for t in T)) for b,ph in atoms) for T in Ts]
    Q=sum(abs(b)**(mp.mpf(1)/s) for b,_ in atoms)
    bonus=abs(A)-mp.mpf(nu)**(1-s)*Q**s
    den_cf=max(sum(abs(BT[i]) for i in f) for f in F); den_1=max(abs(x) for x in BT)
    return len(F),bonus/abs(A),den_cf/abs(A),bonus/den_cf,bonus/den_1
# n=3 two-atom family, s=2, phi=0.3158 (and phi=0)
for s,phi in ((mp.mpf(2),mp.mpf('0.3158')),(mp.mpf(2),mp.mpf(0)),(mp.mpf('1.6'),mp.mpf('0.1969'))):
    c=1/(2*mp.cos(phi)); th=mp.pi/2-phi
    atoms=[(c*mp.e**(-1j*phi),[0,0,0]),(c*mp.e**(1j*phi),[th,th,th])]
    print("n=3 s=%s phi=%s:"%(mp.nstr(s,3),mp.nstr(phi,4)),[mp.nstr(x,8) for x in check(3,s,atoms)])
# n=4 symmetric three-atom family at s=5/2
s=mp.mpf(5)/2; th=mp.mpf('1.013959'); psi=mp.pi-mp.mpf('2.5')*th; c=mp.mpf('0.326610'); d=1-2*c*mp.cos(psi)
atoms=[(d,[0]*4),(c*mp.e**(1j*psi),[th]*4),(c*mp.e**(-1j*psi),[-th]*4)]
print("n=4 s=5/2 sym3:",[mp.nstr(x,8) for x in check(4,s,atoms)]," (d,c,theta,psi)=",[mp.nstr(x,8) for x in (d,c,th,psi)])
