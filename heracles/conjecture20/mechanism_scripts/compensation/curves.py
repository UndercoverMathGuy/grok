# Pi(R)/a_l curves vs gain models, per (p,l).  Usage: python curves.py s progfile [modes]
import json,numpy as np,sys
from game import slope, Pi
s=float(sys.argv[1]); D=[json.loads(l) for l in open(sys.argv[2])]
modes=sys.argv[3].split(',') if len(sys.argv)>3 else ['full']
c1=1-3**(1-s); rmax=3**(s-1)
rhos=np.concatenate([np.geomspace(0.005,0.1,8),np.linspace(0.15,rmax,16)])
res={}
for d in D:
    p=d['p']; S=d['S']; y=dict(zip(S,d['y'])); a={j:3**(1-s)*y[j] for j in S}
    for l in S:
        for m in modes:
            h,mu=slope(p,S,d['Bhalf'],l,a,m)
            pis=[Pi(p,S,d['Bhalf'],l,a,rho*a[l],m)/a[l] for rho in rhos]
            res[f"{p},{l},{m}"]=dict(h=h,rho=list(rhos),pi=pis)
            print(f"p={p} l={l} mode={m} h={h:.4f}  Pi/rho at rho: "+" ".join(f"{rho:.3f}:{v/rho:.3f}" for rho,v in zip(rhos,pis)),flush=True)
json.dump(res,open(f'curves_s{s}_{"_".join(modes)}.json','w'))
