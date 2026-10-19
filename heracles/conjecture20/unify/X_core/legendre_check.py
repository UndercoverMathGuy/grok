# Orchestrator checks on the single-set core RT_n for the Legendre family.
#  U(w) = chi(w) (Legendre symbol) = sum_{l in F} beta_l cos(2 pi l w/p + phi_l), beta_l = 2/sqrt p.
#  (1) LHS = min_delta [sum_l beta_l kappa_l(delta) - max_w |U(w)-U(w-delta)|] = sqrt p - 2.
#  (2) CD_C (exact factorisation at the network's own allocation) needs LHS <= (1-c) sum beta kappa: fails iff c sqrt p > 2.
#  (3) RT_n via the quadratic-residue half: Gamma_s >= (F/2)^{-s} (p - sqrt p)/4 when p = 1 mod 4.
import numpy as np
def legendre(p):
    ch=np.zeros(p); 
    for a in range(1,p): ch[(a*a)%p]=1
    return np.where(ch==1,1.0,-1.0)*(np.arange(p)!=0)
for p in [13,17,29,37,101,127,131,137,149,197,227]:
    F=(p-1)//2; l=np.arange(1,F+1); w=np.arange(p); chi=legendre(p)
    # Fourier coefficients of chi on frequencies 1..F: chi(w) = sum_l 2 Re(c_l e^{2 pi i l w/p})
    c=np.array([np.sum(chi*np.exp(-2j*np.pi*k*w/p))/p for k in l]); beta=2*np.abs(c)
    rec=sum(beta[k]*np.cos(2*np.pi*l[k]*w/p+np.angle(c[k])) for k in range(F))
    assert np.allclose(rec,chi)
    cov=np.array([np.sum(beta*(1-np.cos(2*np.pi*l*d/p))) for d in range(1,p)])
    dep=np.array([np.max(np.abs(chi-np.roll(chi,d))) for d in range(1,p)])
    LHS=np.min(cov-dep)
    out=f"p={p:4d} p%4={p%4} beta_l={beta[0]:.4f}(2/sqrt p={2/np.sqrt(p):.4f}) LHS={LHS:.4f} sqrt(p)-2={np.sqrt(p)-2:.4f}"
    for (n,b) in [(2,2.0),(3,2.0),(10,2.0)]:
        s=(n+1)/b; c_=1-(n/(n+1))**(s-1)
        out+=f" | n={n} c={c_:.3f} CD {'FAILS' if LHS>(1-c_)*np.sqrt(p)+1e-12 else 'ok'}"
    if p%4==1:
        QR={(a*a)%p for a in range(1,p)}; A=[k for k in l if k in QR]
        covA=min(np.sum(1-np.cos(2*np.pi*np.array(A)*d/p)) for d in range(1,p))
        out+=f" | QR-half: |A|={len(A)} min cov={covA:.3f} formula (p-sqrt p)/4={(p-np.sqrt(p))/4:.3f}"
    print(out)
