import sys, mpmath as mp
sys.path.insert(0,'.')
from kstar_exact import eval_arc
n=int(sys.argv[1]); s=(n+1)/2
fr=[float(v) for v in sys.argv[2].split(',')]; cs=[float(v) for v in sys.argv[3].split(',')]
best=None
for f in fr:
    m=max(2,int(round(f*n)))
    row=[]
    for c in cs:
        D=2*float(mp.pi)/(c*n+2)
        d=eval_arc(n,m,D,s,dps=40+n)
        row.append(f"{float(d['ratio']):6.3f}/{float(d['ratio_IFfree']):5.2f}/{float(d['IF']):.2f}")
        if best is None or d['ratio']>best[0]: best=(d['ratio'],m,c,d)
    print(f"n={n} m={m}: "+"  ".join(row),flush=True)
r,m,c,d=best
print(f"BEST n={n}: ratio={mp.nstr(r,6)} m={m} c={c} IFfree={mp.nstr(d['ratio_IFfree'],5)} IF/A={mp.nstr(d['IF'],4)} L/A={mp.nstr(d['L'],5)} E/A2={mp.nstr(d['E'],5)}",flush=True)
