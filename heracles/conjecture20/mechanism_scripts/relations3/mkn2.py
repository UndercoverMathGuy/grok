import sys, numpy as np
TAU=3**0.5; CA=1/(2*3**0.5); C12=1-(2/3)**0.5
G={2003:None,5003:None,10007:float(sys.argv[1]),30011:float(sys.argv[2])}
files={2003:'cert_large_2003.out',5003:'cert_large_5003.out',10007:'cert_large_10007.out',30011:'cert_large_30011_exact.out'}
rows=[]
for p,fn in files.items():
    Gold=None
    for line in open(fn):
        t=line.split()
        if line.startswith('p='): Gold=float(line.split('Gref=')[1].split()[0]); continue
        if not t or not t[0].isdigit(): continue
        K=int(t[1]); e=float(t[2]); cA=float(t[3]); c12=float(t[4]); sk=float(t[5]); sP=float(t[6])
        f=Gold/G[p] if G[p] else 1.0
        e*=f; cA*=f; c12*=f
        rows.append((p,K,e,cA,c12,sk,sP,CA-(1/e-1)/TAU,C12-(1/e-1)/TAU,cA*Gold/f if False else cA*(G[p] or Gold)))
# keep the most spread / most critical rows per p
out=[r'\begin{center}\small\begin{tabular}{rrcccccccc}\toprule',
     r'$p$ & $K$ & $e$ & $\mathrm C_{L2}$(A) & $\mathrm C_{L2}$(P$_{12}$) & $\sum_S\kappa(\delta_0)$ & $\sigma_P$ & need(A) & need(P$_{12}$) & $\Gamma_{\rm crit}$(A)\\\midrule']
for p in files:
    rr=[r for r in rows if r[0]==p]
    best={}
    for r in rr:
        if r[1] not in best or r[3]>best[r[1]][3]: best[r[1]]=r
    for K in sorted(best):
        r=best[K]
        if K<max(best)-8: continue
        out.append(f"{p} & {K} & {r[2]:.3f} & {r[3]:.3f} & {r[4]:.3f} & {r[5]:.1f} & {r[6]:.3f} & {r[7]:.3f} & {r[8]:.3f} & {r[9]:.4f}\\\\")
    out.append(r'\midrule')
out[-1]=r'\bottomrule\end{tabular}\end{center}'
open('n2table.tex','w').write('\n'.join(out)+'\n')
print('\n'.join(out))
