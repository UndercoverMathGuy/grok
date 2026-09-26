"""Generate and independently replay finite aiming-depth certificates.

All accepted LP bounds use exact integer residual repair, with directed interval
trigonometric constants. See HOLES_aiming.md. Local CPU only.
"""
import argparse
from fractions import Fraction as F
from pathlib import Path
import gzip
import hashlib
import itertools
import json
import time
import math
import numpy as np
from scipy.optimize import linprog
from mpmath import iv

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'FINAL_RUNS/HOLES_aiming'
Q=10**30
YQ=10**12
iv.dps=70


def endpoint_fraction(t):
    sign,man,exp,_=t
    return F((-1 if sign else 1)*man)*(F(2)**exp)


def enclose(x):
    a,b=map(endpoint_fraction,x._mpi_)
    lo=(a*Q).numerator//(a*Q).denominator
    bb=b*Q
    hi=-((-bb.numerator)//bb.denominator)
    assert F(lo,Q)<=a<=b<=F(hi,Q)
    return lo,hi


def constants(p,S,L):
    H=[]
    for a in range(p):
        angles=[4*iv.pi*a*k/p for k in S]
        H.append([enclose(iv.cos(t)) for t in angles]+[enclose(iv.sin(t)) for t in angles])
    roots=[(enclose(iv.cos(2*iv.pi*h/L)),enclose(iv.sin(2*iv.pi*h/L))) for h in range(L)]
    cosine=enclose(iv.cos(iv.pi/L))
    return H,roots,cosine


def direction(index,K,L,roots):
    digits=[]
    for _ in range(K-1):
        digits.append(index%L);index//=L
    assert index==0
    pairs=[((Q,Q),(0,0))]+[roots[i] for i in digits]
    return [v[0] for v in pairs]+[v[1] for v in pairs]


def repair(H,d,y):
    assert all(0<=a<len(H) and v>=0 for a,v in y)
    assert len({a for a,v in y})==len(y)
    residual=0
    for j,(dl,du) in enumerate(d):
        lo=dl*YQ-sum(H[a][j][1]*v for a,v in y)
        hi=du*YQ-sum(H[a][j][0]*v for a,v in y)
        assert lo<=hi
        residual+=max(abs(lo),abs(hi))
    return Q*sum(v for a,v in y)+2*residual


def witnesses(H,z):
    # Rational rounding is a definition of the witness, not an error allowance.
    coords=[int(round(float(v)*YQ)) for v in np.r_[z.real,z.imag]]
    numerator=max(sum((hi if v>=0 else lo)*v for (lo,hi),v in zip(row,coords)) for row in H)
    K=len(z);denom=F(0)
    for a,b in zip(coords[:K],coords[K:]):
        value=F(a*a+b*b,YQ**2)
        root=math.isqrt(value.numerator*Q**2//value.denominator)
        lower=F(root,Q)
        assert lower**2<=value<(lower+F(1,Q))**2
        denom+=lower
    assert denom>0
    bound=F(numerator,Q*YQ)/denom
    return {'coordinates':coords,'denominator':YQ,'upper':str(bound),'upper_decimal':float(bound)}


def supports():
    old=json.loads((ROOT/'FINAL_RUNS/P3C/astar_papers.json').read_text())
    result={f"old_{r['p']}":(r['p'],r['S']) for r in old if r['K']==r['K_star']}
    result.update({'ladder_59':(59,[1,6,14,16,19]),'ladder_71':(71,[1,4,11,17,31]),
                   'ladder_113':(113,[1,6,16,31,35,44,55])})
    return result


def replay(path,p,S,L,H,roots,cosine):
    largest=0;count=0
    with gzip.open(path,'rt') as stream:
        for line in stream:
            record=json.loads(line)
            assert record['i']==count
            bound=repair(H,direction(count,len(S),L,roots),record['y'])
            assert bound==record['bound']
            largest=max(largest,bound);count+=1
    assert count==L**(len(S)-1)
    lower=F(cosine[0]*YQ,largest)
    return largest,lower,count


def run(name,L,verify_only):
    p,S=supports()[name];K=len(S)
    assert 3<=L<=p
    OUT.mkdir(parents=True,exist_ok=True)
    tag=f'{name}_L{L}'
    path=OUT/f'{tag}.jsonl.gz';summary_path=OUT/f'{tag}.json'
    H,roots,cosine=constants(p,S,L)
    start=time.time()
    if not verify_only:
        matrix=np.array([[(lo+hi)/(2*Q) for lo,hi in row] for row in H])
        best_ratio=float('inf');best_z=None
        with gzip.open(path,'wt') as stream:
            for index in range(L**(K-1)):
                d=direction(index,K,L,roots)
                objective=np.array([(lo+hi)/(2*Q) for lo,hi in d])
                result=linprog(-objective,A_ub=matrix,b_ub=np.ones(p),bounds=[(None,None)]*(2*K),method='highs')
                assert result.success,result.message
                y=[(a,max(0,int(round(-float(v)*YQ)))) for a,v in enumerate(result.ineqlin.marginals) if v<0]
                y=[(a,v) for a,v in y if v]
                bound=repair(H,d,y)
                stream.write(json.dumps({'i':index,'y':y,'bound':bound},separators=(',',':'))+'\n')
                z=result.x[:K]+1j*result.x[K:]
                ratio=max(matrix@result.x)/sum(abs(z))
                if ratio<best_ratio:best_ratio,best_z=ratio,z.copy()
                if (index+1)%2000==0:
                    print(name,L,index+1,'/',L**(K-1),'seconds',round(time.time()-start,1),flush=True)
        witness=witnesses(H,best_z)
    else:
        witness=json.loads(summary_path.read_text())['upper_witness']
        coords=np.array(witness['coordinates'],dtype=float)/YQ
        # Check the stored rational witness directly, avoiding float re-rounding.
        z=coords[:K]+1j*coords[K:]
        check=witnesses(H,z)
        assert check==witness
    maximum,lower,count=replay(path,p,S,L,H,roots,cosine)
    result={'status':'PASS','name':name,'p':p,'S':S,'K':K,'L':L,'directions_replayed':count,
            'lower':str(lower),'lower_decimal':float(lower),'max_dual_numerator':maximum,
            'dual_bound_denominator':Q*YQ,'cosine_lower':str(F(cosine[0],Q)),
            'upper_witness':witness,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'seconds':time.time()-start}
    summary_path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('upper_witness','max_dual_numerator','dual_bound_denominator','lower')},indent=2),flush=True)
    print('certified upper:',witness['upper_decimal'],flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--names',nargs='+',default=list(supports()))
    parser.add_argument('--L',type=int,default=4)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    for name in args.names:run(name,args.L,args.verify_only)
