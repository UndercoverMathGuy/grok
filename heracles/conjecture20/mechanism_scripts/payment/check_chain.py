# Numerical sanity check of every step of the P12 proof on random configurations with B=C=0.
#  Step 1 (twisted leverage): Q >= L1^r Phi(d'),  d'=|D'|/L1, D'=sum |b| e^{2i phi} xi eta
#  Step 2 (Jensen):           |A| - C|D| <= L1 sqrt(1+C^2-2C d')  for C in [0,c]
#  Step 3 (1-D):              3^{1-s} Phi(d')^s >= sqrt(1+C^2-2C d'), C=min(d',c)
#  Conclusion:                |A| - c|D| <= 3^{1-s} Q^s
import numpy as np
rng=np.random.default_rng(7)
def Phi(x,r): return 0.5*((1+x)**r*(3+x)**(1-r)+(1-x)**r*(3-x)**(1-r))
def project_BC(beta,xi,eta):
    # least-norm complex correction of beta so that sum beta xi = sum beta eta = 0
    M=np.vstack([xi,eta])            # 2 x m
    rhs=M@beta
    corr=M.conj().T@np.linalg.solve(M@M.conj().T,rhs)
    return beta-corr
worst=[np.inf]*4; n=0
for trial in range(200000):
    m=rng.integers(3,8); s=rng.uniform(1.0,1.5); r=1/s; c=1-(2/3)**(s-1)
    if rng.random()<0.3:   # near-diagonal / near-sign structures
        a=rng.choice([0,np.pi/2,np.pi,2*np.pi/3,4*np.pi/3],size=m)+rng.normal(scale=0.05,size=m)
        b=a+rng.normal(scale=0.05,size=m)+rng.uniform(0,2*np.pi)
    else:
        a=rng.uniform(0,2*np.pi,m); b=rng.uniform(0,2*np.pi,m)
    xi=np.exp(1j*a); eta=np.exp(1j*b)
    beta=rng.normal(size=m)+1j*rng.normal(size=m)*rng.choice([0.0,0.1,1.0])
    beta=project_BC(beta,xi,eta)
    if np.abs(beta).sum()<1e-8: continue
    A=beta.sum(); B=(beta*xi).sum(); C_=(beta*eta).sum(); D=(beta*xi*eta).sum()
    assert abs(B)<1e-9 and abs(C_)<1e-9
    rho=np.abs(beta); ph=np.exp(1j*np.angle(beta)); L1=rho.sum(); Q=(rho**r).sum()
    Dp=(rho*ph**2*xi*eta).sum(); dp=min(abs(Dp)/L1,1.0)
    s1=Q-L1**r*Phi(dp,r)
    Cc=min(dp,c)
    s2=L1*np.sqrt(1+Cc**2-2*Cc*dp)-(abs(A)-Cc*abs(D))
    s3=3**(1-s)*Phi(dp,r)**s-np.sqrt(1+Cc**2-2*Cc*dp)
    s4=3**(1-s)*Q**s-(abs(A)-c*abs(D))
    for k,v in enumerate([s1/L1**r,s2/L1,s3,s4/L1]): worst[k]=min(worst[k],v)
    n+=1
print("configs",n)
print("min slack step1 (Q - L1^r Phi)/L1^r      :",worst[0])
print("min slack step2 (Jensen)/L1              :",worst[1])
print("min slack step3 (1-D)                    :",worst[2])
print("min slack P12  (theta Q^s - |A|+c|D|)/L1 :",worst[3])
