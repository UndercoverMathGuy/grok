import numpy as np, sys
sys.path.insert(0,'.')
from kkt import Net
d=np.load(sys.argv[1]); p=int(d['p']); n=int(d['n'])
net=Net(d['U'],d['W'],n,2.0)
A,q,lam,res,h=net.kkt(1e-7)
print('p',p,'n',n,'m',net.m,'h',h,'P',net.cost(),'lam',lam,'res',res)
e=np.array([2*a if a<p/2 else 2*a-p for a in range(p)])
rho=np.exp(2j*np.pi/p)
print('active (N, y, delta, tau, q):')
for (k,dl),qq in zip(A,q):
    N=net.Nc[k]
    print(N, net.y[k], dl, int(N@e), '%.4f'%qq, 'classsize', int(net.cs[k]))
# multiplier uniqueness: null space of J_A
J=np.array([net.grad_gap(k,dl) for k,dl in A]).T
sv=np.linalg.svd(J,compute_uv=False)
print('J_A shape',J.shape,'rank',(sv>1e-9*sv[0]).sum())
# LP: range of each q_k over the multiplier polytope
from scipy.optimize import linprog
gN=net.grad_cost()
Aeq=np.vstack([J,np.ones((1,J.shape[1]))*0])
# normalise: J qt = gN, qt>=0 ; q = qt*lam
rng=[]
for k in range(J.shape[1]):
    c=np.zeros(J.shape[1]); c[k]=1
    lo=linprog(c,A_eq=J,b_eq=gN,bounds=(0,None),method='highs')
    hi=linprog(-c,A_eq=J,b_eq=gN,bounds=(0,None),method='highs')
    rng.append((lo.fun*lam if lo.success else np.nan, -hi.fun*lam if hi.success else np.nan))
print('multiplier ranges (min,max) per active pair:')
print(np.array2string(np.array(rng),precision=4))
# neuron structure
U,W=net.U,net.W
def ft(v,j): return (v*rho**(-j*np.arange(p))).sum()
print('neurons: |u| sign pattern, DC/|u|, freq1/|u|')
r=2/(n+1)
betas=[]
for i in range(net.m):
    u=U[i]; w=W[i]
    C=ft(u,1)**n
    beta=2*p**(-(n+1))*C*np.conj(ft(w,1))
    betas.append(beta)
    print(i, np.sign(u), 'DC %.3f'%(abs(ft(u,0))/np.sqrt(p)/np.linalg.norm(u)), 'wDC %.2e'%abs(ft(w,0)), '|w|/|u| %.4f'%(np.linalg.norm(w)/np.linalg.norm(u)), 'beta', np.round(beta,6), 'arg %.3f'%np.angle(beta))
betas=np.array(betas); Aamp=abs(betas.sum()); Q=(abs(betas)**r).sum(); s=1/r
print('A',Aamp,'sum|beta|',abs(betas).sum(),'IF price nu^{1-s}Q^s',(n+1)**(1-s)*Q**s,'cheapness A/IFprice',Aamp/((n+1)**(1-s)*Q**s))
