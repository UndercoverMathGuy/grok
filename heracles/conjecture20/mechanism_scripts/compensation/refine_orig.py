# Fine-rho refinement of the original (T-blind) game near the two-atom branch.  Two-atom bonus (exact family):
#   G2(rho) = 3 rho - rho^3/c^2 - 1  (rho <= c),  2c - 1 (rho >= c),  2c = (3/2)^{s-1}
import json,numpy as np,sys
from game2 import Game,family
s=float(sys.argv[1]); D={json.loads(l)['p']:json.loads(l) for l in open(sys.argv[2])}
Gs=json.load(open(f'G_sym_s{s}.json')); gr=np.array(sorted(float(k) for k in Gs)); gv=np.array([Gs[k] for k in sorted(Gs,key=float)])
c=0.5*1.5**(s-1)
def G2(rho): return np.where(rho<=c,3*rho-rho**3/c**2-1,2*c-1)
rhos=np.linspace(0.3,3**(s-1),120)
G=np.maximum(np.interp(rhos,gr,gv),G2(rhos))
print('max G/rho interior', (G/rhos).max(), 'at', rhos[np.argmax(G/rhos)])
out=[]
for p,d in sorted(D.items()):
    S=d['S']; y=dict(zip(S,d['y'])); a={j:3**(1-s)*y[j] for j in S}
    for l in S:
        g=Game(p,S,d['Bhalf'],l,a,family(p,S,l,T12=False))
        P=np.array([g.Pi(r*a[l])/a[l] for r in rhos]); v=P/G; i=np.argmin(v)
        out.append((v[i],p,l,rhos[i])); print(f"p={p} l={l} sigma_orig_fine={v[i]:.4f} at rho={rhos[i]:.3f}",flush=True)
out.sort(); print('worst',out[:5])
