import numpy as np, itertools, sys
from prog import solve
for p in [31,101,211,401]:
    s=float(sys.argv[1]); G,y,cov=solve(p,s); S=list(np.nonzero(y)[0]+1)
    # signed relations sum eps_k l_k = 0 mod p with eps in {+-1}, distinct elements, length 2..4
    cnt={}
    for L in (2,3,4):
        c=0
        for sub in itertools.combinations(S,L):
            for eps in itertools.product((1,-1),repeat=L-1):
                if (sub[0]+sum(e*l for e,l in zip(eps,sub[1:])))%p==0: c+=1
        cnt[L]=c
    print(p,"S=",S," relations by length:",cnt,flush=True)
