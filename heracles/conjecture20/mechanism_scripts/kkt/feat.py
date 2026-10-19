import numpy as np, sys
sys.path.insert(0,'.')
from kkt import Net
def features(net, verbose=True):
    p,n=net.p,net.n; nu=n+1; F=range(1,(p-1)//2+1); rho=np.exp(2j*np.pi/p)
    b=net.b; r=b/nu; s=1/r
    ft=lambda v,j:(v*rho**(-j*np.arange(p))).sum()
    out={}
    for j in F:
        bet=np.array([2*p**(-nu)*ft(net.U[i],j)**n*np.conj(ft(net.W[i],j)) for i in range(net.m)])
        A=abs(bet.sum()); Q=(abs(bet)**r).sum(); price=nu**(1-s)*Q**s
        out[j]=(A,Q,price,bet)
        if verbose: print(' j=%d A=%.4e  sum|b|=%.4e  IFprice=%.4e  cheap ratio A/price=%.4f  Re beta=%.4e'%(j,A,abs(bet).sum(),price,A/price if price>0 else np.inf,bet.sum().real))
    return out
if __name__=='__main__':
    d=np.load(sys.argv[1]); net=Net(d['U'],d['W'],int(d['n']),2.0)
    p=net.p; rho=np.exp(2j*np.pi/p); ft=lambda v,j:(v*rho**(-j*np.arange(p))).sum()
    print('margin',net.margin(),'cost',net.cost())
    features(net)
    print('neuron spectra |u^(j)|/||u|| j=0..p-1 and |w^(j)|:')
    for i in range(net.m):
        print(i,'u',np.round([abs(ft(net.U[i],j))/np.sqrt(p)/np.linalg.norm(net.U[i]) for j in range(p)],3),'w',np.round([abs(ft(net.W[i],j))/np.sqrt(p)/np.linalg.norm(net.W[i]) for j in range(p)],3),'sign',np.sign(net.U[i]).astype(int))
