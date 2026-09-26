# Check at p=3: network-level blind certificates (computed from the gap table) equal the population formula
#   b = min{ ReG - |ImG|/sqrt3 , min_{T,zeta in mu_3} Re(G - J_T zeta) } * 3 * (8/... scale)
# neuron i: u=cos(2pi x/3+phi), v=cos(2pi x/3+psi), w=a cos(2pi c/3+chi)
# population: c_i = a e^{i(chi-phi-psi)}/8, v_i=e^{2i phi}, w_i=e^{2i psi}
import numpy as np
rng=np.random.default_rng(0); p=3; x=np.arange(p); mu3=np.exp(2j*np.pi*x/3)
for trial in range(5):
    N=4; phi,psi,chi=rng.uniform(0,2*np.pi,(3,N)); a=rng.random(N)+.2
    E1=np.cos(2*np.pi*x[None]/3+phi[:,None]); E2=np.cos(2*np.pi*x[None]/3+psi[:,None]); W=a[:,None]*np.cos(2*np.pi*x[None]/3+chi[:,None])
    Z=np.einsum('ia,ib,ic->abc',E1,E2,W)
    gap=np.zeros((p,p,p))
    for x1 in x:
        for x2 in x:
            y=(x1+x2)%p; gap[x1,x2]=Z[x1,x2,y]-Z[x1,x2,(y+x)%p]
    clean=[gap[:,:,D].mean() for D in (1,2)]
    inv2=pow(2,-1,p); aims=[]
    for T in ('1','2','12'):
        for s in x:
            tot=0
            for D in (1,2):
                for x1 in x:
                    for x2 in x:
                        val={'1':x1,'2':x2,'12':x1+x2}[T]+D*inv2
                        if val%p==s: tot+=gap[x1,x2,D]
            aims.append(tot/(2*p))                # uniform D over 2 offsets, uniform over the p rows on the line
    net_b=min(clean+aims)
    c=a*np.exp(1j*(chi-phi-psi))/8; v=np.exp(2j*phi); w=np.exp(2j*psi)
    G=c.sum(); J=[(c*v).sum(),(c*w).sum(),(c*v*w).sum()]
    pop_minus=min([G.real-abs(G.imag)/np.sqrt(3)]+[min((G-JT*z).real for z in mu3) for JT in J])
    pop_plus =min([G.real-abs(G.imag)/np.sqrt(3)]+[min((G+JT*z).real for z in mu3) for JT in J])
    print(f"net_b/pop(minus) = {net_b/pop_minus:.6f}    net_b/pop(plus) = {net_b/pop_plus:.6f}")
