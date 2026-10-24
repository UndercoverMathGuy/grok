import numpy as np, sys, json
sys.path.insert(0,'.')
from load import build1d
from polish import polish
from kkt import gamma_IF
sys.path.insert(0,'../..'); sys.dont_write_bytecode=True
from gamma import Gamma
src,out=sys.argv[1],sys.argv[2]; iters=int(sys.argv[3]) if len(sys.argv)>3 else 500
net,p,n=build1d(src,5)
h=net.margin(); c=(1/abs(h))**(1/net.nu); net.U*=c; net.W*=c
s=(n+1)/2; G=Gamma(p,s)
gIF=gamma_IF(p,n,2.0,G)
print('p',p,'n',n,'m',net.m,'Gamma',G,'before',net.gamma()/gIF)
for rep in range(3):
    polish(net,iters=iters)
    h=net.margin(); c=(1/abs(h))**(1/net.nu); net.U*=c; net.W*=c
    A,q,lam,res,hh=net.kkt(1e-7)
    print('round',rep,'ratio',net.gamma()/gIF,'|A|',len(A),'KKT res',res,'lam',lam,'expect',net.nu*hh/(2*net.cost()))
np.savez(out,U=net.U,W=net.W,n=n,p=p)
