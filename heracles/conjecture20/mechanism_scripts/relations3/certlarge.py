import json, sys, numpy as np
sys.path.insert(0,'.')
from cert import certify
p=int(sys.argv[1]); d=json.load(open(f'large_{p}.json')); runs=d['runs']; Gref=max(r['Gamma'] for r in runs)
print(f"p={p} Gref={Gref:.5f} lnp={np.log(p):.2f}")
print("  K  |S|   e     CL2_A  CL2_12  sumk  sigP  tiltV  sigT | reqA  req12 | Thm1(A,P/T)  Thm1(12,P/T)")
for r in sorted(runs,key=lambda r:(r['K'],-r['Gamma'])):
    R=certify(p,r['S'],r['y'],Gref,binned=(4000 if p>20000 else None),evals=int(sys.argv[2]) if len(sys.argv)>2 else 1200)
    okA=('P' if R['sigP']>=R['req_A'] else '-')+('T' if R['sigT']>=R['req_A'] else '-')
    ok12=('P' if R['sigP']>=R['req_12'] else '-')+('T' if R['sigT']>=R['req_12'] else '-')
    print(f"{r['K']:3d} {R['K']:3d} {R['e']:6.4f} {R['CL2_A']:6.4f} {R['CL2_12']:6.4f} {R['sumk']:5.2f} {R['sigP']:5.3f} {R['tiltV']:5.3f} {R['sigT']:5.3f} | {R['req_A']:5.3f} {R['req_12']:5.3f} |   {okA}          {ok12}",flush=True)
