# sigma (Riesz, separate costs) at y* with: mu on B (task), mu on all directions (coverage term), mu fixed to the KKT measure mu*.
import json,numpy as np,sys
from alloc import sigma_y, load_grid
s=float(sys.argv[1]); ax,G=load_grid(s)
for line in open('prog15.out' if s==1.5 else 'prog12.out'):
    d=json.loads(line); p=d['p']; S=d['S']; yv=np.array(d['y']); F=(p-1)//2
    mu={dd:lam for dd,lam in zip(d['Bhalf'],d['lam'])}
    for l in S:
        b=sigma_y(p,s,S,yv,l,d['Gamma'],ax,G,Dhalf=d['Bhalf'])[0]
        al=sigma_y(p,s,S,yv,l,d['Gamma'],ax,G)[0]
        ms=sigma_y(p,s,S,yv,l,d['Gamma'],ax,G,mu_dir_half=mu,Dhalf=d['Bhalf'])[0]
        print(f"p={p} l={l} sigmaB={b[0]:.3f} sigmaAll={al[0]:.3f} sigmaMuStar={ms[0]:.3f} at {ms[1]}",flush=True)
