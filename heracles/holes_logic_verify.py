"""Independent fp64 checks of HOLES_logic.md (not the exact certificate).

Checks the new analytic bounds against quadrature; checks atom lemmas against
explicit populations; checks the profile, density and normalization with full
logit tables. Writes only FINAL_RUNS/HOLES_logic/verification.json.
"""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, linprog

OUT = Path(__file__).resolve().parent / "FINAL_RUNS/HOLES_logic"
RNG = np.random.default_rng(20260922)
r = 2/3


def coeffs(c, V):
    return np.array([np.sum(c*np.prod(V[:, [j for j in range(V.shape[1]) if t >> j & 1]], axis=1))
                     for t in range(1 << V.shape[1])])


def aimed(C, depth=.5):
    def f(phi):
        t = depth*np.exp(1j*phi)
        return (C[0]+C[1]*t).real-depth*abs(C[2]+C[3]*t)
    grid = np.linspace(0, 2*np.pi, 65)
    candidates = [minimize_scalar(f, bounds=(a,b), method="bounded",
                                 options={"xatol": 1e-13}) for a,b in zip(grid[:-1],grid[1:])]
    result = min(candidates, key=lambda z:z.fun)
    t1 = depth*np.exp(1j*result.x)
    z = C[2]+C[3]*t1
    t2 = -depth*np.conj(z)/abs(z) if abs(z)>0 else depth
    return result.fun, (t1,t2)


def atom_checks():
    worst = {"cost_minus_tool": float("inf"), "theorem_ratio": float("inf"),
             "D1_slack": float("inf"), "D2_slack": float("inf")}
    positive = 0
    for it in range(500):
        N = int(RNG.integers(3,10))
        c = RNG.normal(size=N)+1j*RNG.normal(size=N)
        V = np.exp(2j*np.pi*RNG.random((N,2)))
        if it % 2 == 0:
            V[:3] = np.exp(2j*np.pi*np.arange(3)[:,None]/3)*np.exp(.08j*RNG.normal(size=(3,2)))
            c[:3] = 1/3+.015*(RNG.normal(size=3)+1j*RNG.normal(size=3))
            c[3:] *= .002
        C = coeffs(c,V)
        c /= abs(C[0]); C = coeffs(c,V)
        m,_ = aimed(C)
        S = np.abs(c).sum(); w = np.sqrt(max(0,S*S-1))
        d = 1-m; cost = np.sum(abs(c)**r)
        if m<=0:
            continue
        positive += 1
        worst["theorem_ratio"] = min(worst["theorem_ratio"], cost/(3**(1/3)*m**r))
        worst["D1_slack"] = min(worst["D1_slack"], *(d-.5**t.bit_count()*abs(C[t]) for t in range(1,4)))
        U = coeffs(c,np.conj(V))
        worst["D2_slack"] = min(worst["D2_slack"], *(abs(C[t])+2*w-abs(U[t]) for t in range(1,4)))
        u,v = min(S,2*d+2*w),min(S,4*d+2*w)
        a2=max(0,3-4*d*u-4*d*v)/9
        a3=max(0,1-4*d*u-4*d*v-4*d*d*v-4*d*u*u)/27
        b1=max(0,(m-S/4)/2); b2=max(0,2*(m-3*S/4)/3)
        peel=(S-b1)**r+(b1-b2)**r+b2**r
        L2=2*(S/2)**(4/9)*a2**(1/9)
        L3=3*(S/3)**(10/27)*(a2/3)**(2/27)*a3**(4/81)
        worst["cost_minus_tool"] = min(worst["cost_minus_tool"],cost-max(peel,L2,L3,S**r))
    assert positive>=250 and min(worst.values())>=-1e-9, worst
    return {"populations":500,"positive_aimed_output":positive,**worst}


def quadrature_checks():
    worst=float("inf")
    for m in (2,3):
        for _ in range(100):
            a=np.exp(RNG.uniform(-5,3,m))
            # t=u^3 and integration by parts: c_r/r * 3 integral P'(u^3)/P(u^3) du.
            def integrand(u):
                t=u**3
                return sum((j+1)*a[j]*t**j for j in range(m))/(1+sum(a[j]*t**(j+1) for j in range(m)))
            integral,error=quad(integrand,0,np.inf,epsabs=1e-10,epsrel=1e-10)
            value=3*np.sin(np.pi*r)/np.pi*integral
            lower=(2*(a[0]/2)**(4/9)*a[1]**(1/9) if m==2 else
                   3*(a[0]/3)**(10/27)*(a[1]/3)**(2/27)*a[2]**(4/81))
            assert lower<=value+1e-8,(a,lower,value,error)
            worst=min(worst,value/lower)
    # Exact equality occurs for P=(1+alpha*t)^m.
    for m in (2,3):
        alpha=.7
        lower=(2*(2*alpha/2)**(4/9)*(alpha**2)**(1/9) if m==2 else
               3*(3*alpha/3)**(10/27)*(3*alpha**2/3)**(2/27)*(alpha**3)**(4/81))
        assert abs(lower-m*alpha**r)<1e-14
    return {"polynomials":200,"minimum_integral_over_bound":worst,"equality_checks":2}


def network_checks():
    profile_err=0.; density_err=0.; lp_err=0.; block_err=0.; cells=0
    for n in (2,3):
        for p in (5,7,11):
            for trial in range(4):
                rows=np.indices((p,)*n).reshape(n,-1).T
                labels=rows.sum(axis=1)%p
                z=np.zeros((len(rows),p)); prof=np.zeros_like(z)
                Cmap={}; penalty=0.; atomcost=0.
                for k in (1,2):
                    N=3
                    alpha=RNG.uniform(0,2*np.pi,(N,n)); beta=RNG.uniform(0,2*np.pi,N)
                    amplitudes=RNG.uniform(.1,1.5,(N,n+1))
                    c=amplitudes.prod(axis=1)*np.exp(1j*(alpha.sum(axis=1)-beta))
                    V=np.exp(-2j*alpha); C=coeffs(c,V); Cmap[k]=C
                    atomcost+=sum(abs(c)**(2/(n+1)))
                    penalty+=p/2*np.sum(amplitudes**2)
                    for i in range(N):
                        act=np.prod(amplitudes[i,:n]*np.cos(2*np.pi*k*rows/p+alpha[i]),axis=1)
                        z+=act[:,None]*(amplitudes[i,n]*np.cos(2*np.pi*k*np.arange(p)/p+beta[i]))
                    h=np.zeros(len(rows),complex)
                    for t in range(1<<n):
                        sigma=sum((rows[:,j] for j in range(n) if t>>j&1),np.zeros(len(rows),int))
                        h+=C[t]*np.exp(-4j*np.pi*k*sigma/p)
                    prof+=np.real(h[:,None]*(1-np.exp(-2j*np.pi*k*np.arange(p)/p)))
                gap=z[np.arange(len(rows)),labels,None]-z[np.arange(len(rows))[:,None],(labels[:,None]+np.arange(p))%p]
                profile_err=max(profile_err,float(np.max(abs(2**n*gap-prof))))
                assert penalty+1e-10>=((n+1)*p/2)*atomcost
                for partition in ([[j] for j in range(n)],[list(range(n))]):
                    density=np.ones(len(rows)); predicted=np.zeros(p)
                    moments=[]
                    for block in partition:
                        nu=RNG.random(p); nu/=nu.sum()
                        density*=nu[rows[:,block].sum(axis=1)%p]/p**(len(block)-1)
                        moments.append({k:sum(nu*np.exp(-4j*np.pi*k*np.arange(p)/p)) for k in Cmap})
                    assert abs(density.sum()-1)<1e-13
                    for k,C in Cmap.items():
                        g=0j
                        for subset in range(1<<len(partition)):
                            mask=0; prod=1+0j
                            for i,block in enumerate(partition):
                                if subset>>i&1:
                                    mask|=sum(1<<j for j in block); prod*=moments[i][k]
                            g+=C[mask]*prod
                        predicted+=np.real((1-np.exp(-2j*np.pi*k*np.arange(p)/p))*g)
                    density_err=max(density_err,float(np.max(abs(density@prof-predicted))))
                    assert np.min(prof[:,1:])<=np.min((predicted[1:]+predicted[:0:-1])/2)+1e-10
                cells+=1
    # Arbitrary phases simultaneously realized on the known half-depth support.
    p=11; support=(1,4)
    for _ in range(20):
        phases=RNG.uniform(0,2*np.pi,2)
        mat=[np.ones(p)]; rhs=[1.]
        for k,phi in zip(support,phases):
            character=np.exp(-4j*np.pi*k*np.arange(p)/p)
            mat.extend((character.real,character.imag)); rhs.extend((.5*np.cos(phi),.5*np.sin(phi)))
        result=linprog(np.zeros(p),A_eq=mat,b_eq=rhs,bounds=(0,None),method="highs")
        assert result.success
        lp_err=max(lp_err,float(np.max(abs(np.array(mat)@result.x-rhs))))
    for p in (3,5,7,11,17,31):
        rows=np.indices((p,p)).reshape(2,-1).T; labels=rows.sum(axis=1)%p
        z=np.zeros((p*p,p)); penalty=0.
        for l in range(3):
            alpha=np.pi*l/3
            act=np.cos(2*np.pi*rows[:,0]/p+alpha)*np.cos(2*np.pi*rows[:,1]/p+alpha)
            z+=act[:,None]*np.cos(2*np.pi*np.arange(p)/p+2*alpha)
            penalty+=3*p/2
        gaps=z[np.arange(p*p),labels,None]-z
        gaps[np.arange(p*p),labels]=np.inf
        gamma=gaps.min()/penalty**1.5
        target=3**(-.5)*2*(6*p)**(-1.5)*(1-np.cos(2*np.pi/p))
        block_err=max(block_err,abs(gamma/target-1))
    assert profile_err<1e-10 and density_err<1e-10 and lp_err<1e-10 and block_err<1e-10
    return {"full_table_networks":cells,"profile_max_error":profile_err,"density_max_error":density_err,
            "moment_LPs":20,"moment_max_error":lp_err,"single_frequency_block_cells":6,
            "block_max_relative_error":block_err}


if __name__ == "__main__":
    result={"quadrature":quadrature_checks(),"atoms":atom_checks(),"networks":network_checks(),"status":"PASS"}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
