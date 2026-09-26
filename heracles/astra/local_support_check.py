"""Exhaustive finite check of the Fourier-support counting lemma at small primes."""
import json
from pathlib import Path
records=[]
for p in (3,5,7,11,13,17):
 h=(p-1)//2;checked=0;largest=0
 for mask in range(1,1<<h):
  S={k+1 for k in range(h) if mask>>k&1};K=len(S)
  for u in range(p):
   for v in range(p):
    candidates=[]
    for c in range(1,p):
     a=(u-c)%p;b=(v-c)%p
     aa=min(a,(-a)%p);bb=min(b,(-b)%p);cc=min(c,p-c)
     if (aa==bb and aa in S) or (aa==cc and aa in S) or (bb==cc and bb in S):
      candidates.append(c)
    assert len(candidates)<=2*K,(p,S,u,v,candidates)
    checked+=1;largest=max(largest,len(candidates))
  if p==17 and mask%64==0:print({'p':p,'supports_checked':mask},flush=True)
 records.append({'p':p,'supports':(1<<h)-1,'mode_support_pairs':checked,'maximum_output_support':largest})
 print(records[-1],flush=True)
Path(__file__).with_name('local_support_check.json').write_text(json.dumps(records,indent=2)+'\n')
