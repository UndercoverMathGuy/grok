"""Independent fp64 checks and an interval-verified scalar-route obstruction."""
import json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from mpmath import iv
from holes_logic_verify import aimed,coeffs

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'FINAL_RUNS/HOLES_aiming'
RNG=np.random.default_rng(92871)
iv.dps=60


def scalar_obstruction():
    A=iv.mpf(7)/20;lam=iv.mpf(3)/5;d=iv.mpf(11)/160;w=iv.mpf(7)/20;t=iv.mpf(1)/3
    S=iv.sqrt(1+w*w);mu=1-d;r=iv.mpf(2)/3
    def mn(a,b):
        if a.b<=b.a:return a
        if b.b<=a.a:return b
        raise ArithmeticError('ambiguous minimum')
    def mx(a,b):
        if a.a>=b.b:return a
        if b.a>=a.b:return b
        raise ArithmeticError('ambiguous maximum')
    zero=iv.mpf(0)
    D=d/A;L=mn(2*D,D+A*t);v=mn(S,t+2*w)
    H=mn(S*L,D*D+(L-D)**2+2*w*L)+t*v
    a2=mx(zero,3-H)/9
    a3=mx(zero,1-H-(L/2)**2*v-mn(S*S,(L/2+2*w)**2)*t)/27
    b1=mx(mx(zero,(mu-S*(1-A)**2)/(4*A)),(S-t)/2)
    b2=mx(zero,(mu-S*(1-A*A))/(2*A*(1+A)))
    peel=(S-b1)**r+(b1-b2)**r+b2**r
    c2=2*(S/2)**(iv.mpf(4)/9)*a2**(iv.mpf(1)/9)
    c3=3*(S/3)**(iv.mpf(10)/27)*(a2/3)**(iv.mpf(2)/27)*a3**(iv.mpf(4)/81)
    target=3**(iv.mpf(1)/3)*(lam*mu**r+(1-lam)*(1-A*t)**r)
    tools={'trivial':S**r,'peel':peel,'quadratic':c2,'cubic':c3}
    assert all(value.b<target.a for value in tools.values())
    least_mixture_target=3**(iv.mpf(1)/3)*mn(mu**r,(1-A*t)**r)
    assert all(value.b<least_mixture_target.a for value in tools.values())
    # Check the retained scalar admissibility constraints as well.
    assert t.b<S.a and (A*A*t).b<d.a
    return {'A':'7/20','lambda':'3/5','d':'11/160','w':'7/20','t':'1/3',
            'tools':{k:str(v) for k,v in tools.items()},'target':str(target),
            'target_lower_for_every_mixture_weight':str(least_mixture_target),
            'ratio_upper':max(float(v.b) for v in tools.values())/float(target.a),
            'meaning':'The scalar sufficient bounds fail; no atom counterexample is asserted.'}


def depth_checks():
    A=.47;r=2/3;minimum=float('inf');slack=float('inf');count=0
    for i in range(250):
        N=int(RNG.integers(3,10));c=RNG.normal(size=N)+1j*RNG.normal(size=N)
        V=np.exp(2j*np.pi*RNG.random((N,2)))
        if i%2==0:
            c[:3]=1/3+.01*(RNG.normal(size=3)+1j*RNG.normal(size=3));c[3:]*=.002
            V[:3]=np.exp(2j*np.pi*np.arange(3)[:,None]/3)*np.exp(.05j*RNG.normal(size=(3,2)))
        C=coeffs(c,V);c/=abs(C[0]);C=coeffs(c,V)
        m,_=aimed(C,A)
        if m<=0:continue
        count+=1;S=sum(abs(c));w=np.sqrt(max(0,S*S-1));d=1-m;D=d/A;T=d/A**2
        u=min(S,D+2*w);v=min(S,T+2*w)
        a2=max(0,3-2*D*u-T*v)/9;a3=max(0,1-2*D*u-T*v-D*D*v-u*u*T)/27
        b1=max(0,(m-S*(1-A)**2)/(4*A));b2=max(0,(m-S*(1-A*A))/(2*A*(1+A)))
        tools=[S**r,(S-b1)**r+(b1-b2)**r+b2**r,2*(S/2)**(4/9)*a2**(1/9),
               3*(S/3)**(10/27)*(a2/3)**(2/27)*a3**(4/81)]
        cost=sum(abs(c)**r);slack=min(slack,cost-max(tools));minimum=min(minimum,cost/(3**(1/3)*m**r))
    assert minimum>=1-1e-10 and slack>=-1e-10
    return {'populations':250,'positive_output':count,'minimum_theorem_ratio':minimum,'minimum_tool_slack':slack}


def support_checks():
    names=['old_59_L8','old_71_L4','old_97_L6','old_113_L6','old_127_L6',
           'ladder_59_L6','ladder_71_L16','ladder_113_L6']
    results=[]
    for name in names:
        rec=json.loads((OUT/(name+'.json')).read_text());p=rec['p'];S=rec['S'];K=len(S)
        assert F(rec['lower'])>F(7,20)
        if name=='ladder_71_L16':
            assert F(rec['lower'])>F(47,100)
        E=np.exp(-4j*np.pi*np.outer(np.arange(p),S)/p)
        witness=rec['upper_witness'];v=np.array(witness['coordinates'])/witness['denominator']
        z=v[:K]+1j*v[K:];ratio=np.max((E@z).real)/sum(abs(z))
        assert abs(ratio-witness['upper_decimal'])<1e-10
        z=RNG.normal(size=(1000,K))+1j*RNG.normal(size=(1000,K))
        sampled=np.max((E@z.T).real,axis=0)/np.sum(abs(z),axis=1)
        assert sampled.min()>=rec['lower_decimal']-1e-12
        results.append({'name':name,'lower':rec['lower_decimal'],'upper':witness['upper_decimal'],
                        'witness_fp64':ratio,'minimum_random_ratio':float(sampled.min())})
    # Moment feasibility at the actual covered paper support, not just old support labels.
    p=71;S=[1,4,11,17,31];E=np.exp(-4j*np.pi*np.outer(S,np.arange(p))/p)
    matrix=np.r_[np.ones((1,p)),E.real,E.imag];max_error=0.
    for _ in range(30):
        target=.47*np.exp(2j*np.pi*RNG.random(len(S)))
        rhs=np.r_[1.,target.real,target.imag]
        result=linprog(np.zeros(p),A_eq=matrix,b_eq=rhs,bounds=(0,None),method='highs')
        assert result.success
        max_error=max(max_error,float(np.max(abs(matrix@result.x-rhs))))
    old={1,7,12,16,34};new={1,4,11,17,31}
    assert {min((31*k)%71,71-(31*k)%71) for k in old}==new
    record=json.loads((OUT/'ladder_113_L6.json').read_text())
    assert F(record['upper_witness']['upper'])<F(423,1000)
    # The M2 population has h=1+t1*t2 and singleton score sqrt(3/2)*(1-A^2).
    obstruction_score_squared=F(3,2)*(1-F(423,1000)**2)**2
    assert obstruction_score_squared>1
    return {'supports':results,'moment_LPs':30,'moment_max_error':max_error,
            'p71_frequency_multiplier':31,
            'p113_M2_singleton_score_squared_at_A_0_423':str(obstruction_score_squared)}


if __name__=='__main__':
    result={'obstruction':scalar_obstruction(),'depth47':depth_checks(),'supports':support_checks(),'status':'PASS'}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
