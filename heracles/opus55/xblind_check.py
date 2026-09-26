# Check: the "half-offset" test distributions are blind to every Fourier component of the logit
# tensor except same-|k| ones (|a|=|b|=|c|), i.e. to all cross-frequency (impure) content.
import numpy as np
rng=np.random.default_rng(0)
def gap_table(Z,p):
    x=np.arange(p); G=np.zeros((p,p,p))   # G[x1,x2,D]
    for x1 in range(p):
        for x2 in range(p):
            y=(x1+x2)%p
            G[x1,x2]=Z[x1,x2,y]-Z[x1,x2,(y+x)%p]
    return G
def same_k_projection(Z,p):
    F=np.fft.fftn(Z); a=np.arange(p); f=np.minimum(a,p-a)
    keep=(f[:,None,None]==f[None,:,None])&(f[None,:,None]==f[None,None,:])&(f[None,None,:]>0)
    return np.real(np.fft.ifftn(F*keep))
def E_pi(G,rho,T,p):
    inv2=pow(2,-1,p); tot=0
    for D in range(1,p):
        for x1 in range(p):
            for x2 in range(p):
                s={'1':x1,'2':x2,'12':x1+x2}[T]+D*inv2
                tot+=rho[s%p]*G[x1,x2,D]/p          # (other coordinate uniform) weight rho/p
    return tot/(p-1)
for p in (5,7,11):
    Z=rng.standard_normal((p,p,p)); Zs=same_k_projection(Z,p)
    G,Gs=gap_table(Z,p),gap_table(Zs,p)
    for T in ('1','2','12'):
        rho=rng.random(p); rho/=rho.sum()
        a,b=E_pi(G,rho,T,p),E_pi(Gs,rho,T,p)
        print(f"p={p} type {T:>2}: full tensor {a:+.12f}   same-|k| part only {b:+.12f}   diff {a-b:.1e}")
    # control: a product density (not half-offset-locked) is NOT blind
    rho1=rng.random(p);rho1/=rho1.sum()
    ctrl=lambda G: sum(rho1[x1]*G[x1,x2,D] for D in range(1,p) for x1 in range(p) for x2 in range(p))/(p*(p-1))
    print(f"p={p} control product density: full {ctrl(G):+.6f}  same-|k| {ctrl(Gs):+.6f}")
