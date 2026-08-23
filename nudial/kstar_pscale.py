# p-scaling of M_K = max_S min_Delta sum_{k in S}(1-cos 2pi k Delta/p); local search with restarts.
import numpy as np, json, sys, time
rng = np.random.default_rng(0)
def G(p):
    k = np.arange(1, (p-1)//2+1); d = np.arange(1, p)
    return (1 - np.cos(2*np.pi*np.outer(k, d)/p)).astype(np.float32)
def best_subset(g, K, restarts):
    nf = g.shape[0]; others = np.arange(1, nf); best=-1; bestS=None
    for r in range(restarts):
        S = set([0]) | set(rng.choice(others, K-1, replace=False).tolist())
        cur = g[list(S)].sum(0).min(); improved=True
        while improved:
            improved=False
            for out in list(S):
                if out==0: continue
                base = g[[x for x in S if x!=out]].sum(0)
                cand = (base[None,:]+g).min(1); cand[list(S)] = -1
                j = int(cand.argmax())
                if cand[j] > cur+1e-6:
                    S.remove(out); S.add(j); cur=cand[j]; improved=True; break
        if cur>best: best,bestS=float(cur),tuple(sorted(x+1 for x in S))
    return best,bestS
hermite = {2:2/3**.5, 3:2**(1/3), 4:2**.5, 5:8**.2, 6:(64/3)**(1/6), 7:64**(1/7), 8:2.0}
out={}
for p in [int(x) for x in sys.argv[1:]]:
    g=G(p); t0=time.time(); rows={}
    for K in range(1,13):
        if K==1: M,S=float(g[0].min()),(1,)
        else: M,S=best_subset(g,K, 40 if p<=1009 else 15)
        rows[K]=dict(M=M,S=S,c=K-M, herm=(2*np.pi**2*hermite.get(K,K/(2*np.pi*np.e)*1.0)*p**(-2/K)) if K>=2 else None)
        print(p,K,f"M={M:.4f} c={K-M:.3f} hermite={rows[K]['herm']}",S,f"{time.time()-t0:.0f}s",flush=True)
    for nu in (3,4,5,6):
        mg={K:K**(-nu/2)*r['M'] for K,r in rows.items()}
        Kd=(p-1)//2; mg_dense=Kd**(-nu/2)*(p/2)
        ks=max(mg,key=mg.get)
        print(f"  p={p} nu={nu}: K*={ks} margin={mg[ks]:.4g} runner-up={sorted(mg,key=mg.get)[-2]} ({mg[sorted(mg,key=mg.get)[-2]]/mg[ks]:.3f}) dense={mg_dense:.3g}",flush=True)
    out[p]=rows
json.dump(out,open('pscale.json','w'),indent=1,default=str)
