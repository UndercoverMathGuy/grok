# n-operand generalisation of astra's Fourier support-counting lemma (LOCAL_THEOREM s4):
# for every input mode u in Z_p^n, the number of readout frequencies c != 0 at which some
# first-order neuron term can live (n of the n+1 slot frequencies share one |k|, k in S) is <= 2K.
import itertools
def fold(a,p): a%=p; return min(a,p-a)
def run(n,p):
    h=(p-1)//2; worst={}
    for K in range(1,h+1):
        for S in itertools.combinations(range(1,h+1),K):
            if 1 not in S: continue  # dilation symmetry
            Sset=set(S)
            for u in itertools.product(range(p),repeat=n):
                cnt=0
                for c in range(1,p):
                    fr=[fold(uj-c,p) for uj in u]+[fold(c,p)]
                    ok=False
                    for s in range(n+1):
                        rest=[fr[i] for i in range(n+1) if i!=s]
                        if rest[0] in Sset and all(f==rest[0] for f in rest): ok=True;break
                    cnt+=ok
                if cnt>2*K: worst.setdefault(K,[]).append((S,u,cnt))
    return worst
for n,p in [(2,11),(3,5),(3,7),(3,11),(4,5),(4,7)]:
    w=run(n,p)
    print(n,p,{K:(len(v),v[:2]) for K,v in w.items()} or 'OK <=2K everywhere',flush=True)
