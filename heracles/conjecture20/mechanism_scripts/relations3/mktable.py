# Snapshot of programme supports (s=1.5): best Gamma per p from relations2 json and sibling ideaC runs; small p by exhaustive search.
import json, glob, sys
sys.path.insert(0,'/home/user/grok/heracles/conjecture20/mechanism_scripts')
out={}
fns=glob.glob('gam_s1.5_*.json')+['../relations2/supports_s1.5.json']
for fn in fns:
    d=json.load(open(fn))
    for p,v in d.items():
        if p not in out or v['Gamma']>out[p]['Gamma']: out[p]={k:v[k] for k in ['Gamma','S','y']}
from progx import exhaustive
import numpy as np
for p in [5,7,11,13,17,19,23,29]:
    G,S,y=exhaustive(p,1.5,min((p-1)//2,4))
    out[str(p)]={'Gamma':float(G),'S':[int(a) for a in S],'y':[float(a) for a in y]}
    print(p,G,S,flush=True)
json.dump(out,open('supports.json','w'),indent=0)
print(sorted(map(int,out)))
