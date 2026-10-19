import numpy as np, sys
sys.path.insert(0,'.')
from kkt import Net
from polish import polish
d=np.load(sys.argv[1]); eps=float(sys.argv[2]); T=int(sys.argv[3])
base=Net(d['U'],d['W'],int(d['n']),2.0); g0=base.gamma()
rng=np.random.default_rng(1)
for t in range(T):
    net=Net(d['U']*(1+eps*rng.standard_normal(d['U'].shape)),d['W']*(1+eps*rng.standard_normal(d['W'].shape)),int(d['n']),2.0)
    gp=net.gamma()
    h=net.margin(); 
    if h<=0: print('perturbed margin <=0, skip'); continue
    c=(1/h)**(1/net.nu); net.U*=c; net.W*=c
    polish(net,iters=300,verbose=False)
    print('trial',t,'perturbed %.6f -> repolished %.10f (base 1)'%(gp/g0, net.gamma()/g0))
