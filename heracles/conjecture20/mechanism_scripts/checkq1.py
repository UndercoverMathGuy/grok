# Sanity check of Theorem q1n2: min_x cm(Phi) <= Re beta_1 - cos(pi/p) max_T |beta_T| for random single-feature n=2 networks.
import numpy as np
rng=np.random.default_rng(0); worst=-np.inf
for p in [3,5,7,11]:
    rho=np.exp(2j*np.pi/p); a=np.arange(p); th=np.pi/p
    for trial in range(3000):
        m=rng.integers(1,7)
        U1=rng.standard_normal((m,p))*rng.uniform(0,1,(m,1)); U2=rng.standard_normal((m,p))
        if trial%3==0: U1=np.cos(2*np.pi*a/p+rng.uniform(0,6.3,(m,1)))+0.3*rng.standard_normal((m,p))*rng.uniform(0,1)
        W=rng.standard_normal(m)+1j*rng.standard_normal(m)           # hat w_i(1) (frequency-1 outputs)
        V1=U1*rho**a; V2=U2*rho**a
        Phi=(2/p)*np.einsum('i,ia,ib->ab',W,V1,V2)                   # Phi_1(x1,x2)
        cm=np.real(Phi)-abs(np.imag(Phi))/np.tan(th)
        A=Phi.mean()                                                   # = conj(beta_1)
        X1,X2=np.meshgrid(a,a,indexing='ij')
        BT=[abs((Phi*rho**(-2*X1)).mean()),abs((Phi*rho**(-2*X2)).mean()),abs((Phi*rho**(-2*(X1+X2))).mean())]
        lhs=cm.min(); rhs=np.real(A)-np.cos(th)*max(BT)
        worst=max(worst,lhs-rhs)
    print(p,"max over trials of [min cm - (Re A - cos(pi/p) max|B_T|)] =",worst)
