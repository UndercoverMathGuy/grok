# Refine a support by swap/add/drop moves on the restricted programme value (reweighted LP); targeted candidates
# = frequencies with large kappa at the current binding directions.
import numpy as np, sys, json, time
sys.path.insert(0,'.')
from largep import restricted, kap_rows
def refine(p,S,G,rng,rounds=4,ncand=30,tmax=1500):
    F=(p-1)//2; d=np.arange(1,F+1); t0=time.time(); S=list(S)
    for rd in range(rounds):
        improved=False
        Gs,S2,y=restricted(p,S); cov=kap_rows(p,S2,d).T@np.array(y); B=d[cov<=cov.min()*(1+1e-6)]
        # targeted candidates: maximise sum over binding directions of kappa
        allc=np.arange(1,F+1); allc=allc[~np.isin(allc,S)]
        sc=np.zeros(len(allc))
        for b in B: sc+=1-np.cos(2*np.pi*allc*b/p)
        cands=list(allc[np.argsort(-sc)[:ncand]])+list(rng.choice(allc,ncand,replace=False))
        moves=[]
        for c in cands:
            moves.append(sorted(S+[int(c)]))
            for j in range(len(S)): moves.append(sorted(S[:j]+S[j+1:]+[int(c)]))
        for j in range(len(S)): moves.append(sorted(S[:j]+S[j+1:]))
        rng.shuffle(moves)
        for M in moves:
            if time.time()-t0>tmax: return G,S
            Gm,Sm,_=restricted(p,M,iters=40)
            if Gm>G*(1+1e-7): G,S=Gm,list(Sm); improved=True; print(f"  improved {G:.5f} K={len(S)}",flush=True); break
        if not improved and rd>0: break
    return G,S
if __name__=="__main__":
    p=int(sys.argv[1]); tmax=float(sys.argv[2]); d=json.load(open(f'large_{p}.json')); rng=np.random.default_rng(7)
    runs=sorted(d['runs'],key=lambda r:-r['Gamma'])[:int(sys.argv[3])]
    for r in runs:
        G,S=refine(p,r['S'],r['Gamma'],rng,rounds=50,tmax=tmax)
        Gs,S2,y=restricted(p,S)
        print(f"p={p} start K={len(r['S'])} G={r['Gamma']:.5f} -> K={len(S2)} G={Gs:.5f}",flush=True)
        d['runs'].append({'K':len(S2),'rep':'refined','Gamma':Gs,'S':S2,'y':y})
        json.dump(d,open(f'large_{p}.json','w'))
