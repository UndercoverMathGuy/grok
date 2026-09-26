import numpy as np
rng=np.random.default_rng(0)
for p in [5,11,23]:
  for n in [1,2,3]:
    best=0
    for trial in range(200):
      a=np.arange(p)
      if trial==0:
        us=[np.cos(2*np.pi*a/p) for _ in range(n)]
      else:
        us=[rng.standard_normal(p) for _ in range(n)]
      grids=np.meshgrid(*[a]*n,indexing='ij')
      S=sum(us[t][grids[t]] for t in range(n))
      h=np.maximum(S,0)
      # diagonal coefficients: hat h(j,...,j)=sum_x h(x) rho^{-j sum a}
      s=sum(grids)%p
      G=np.array([np.sum(h*np.exp(-2j*np.pi*j*s/p)) for j in range(p)])
      # choose optimal w: maximize sum_{j in F}|G_j||w_hat_j| / (||u||^2+||w||^2)
      F=range(1,(p-1)//2+1)
      # w_hat_j ∝ G_j over F (Cauchy-Schwarz optimum); ||w||^2 = (2/p) sum_F |w_hat_j|^2
      g=np.array([abs(G[j]) for j in F])
      # sum g_j c_j subject to (2/p) sum c_j^2 = W2 -> max = sqrt(p W2/2)*||g||
      U2=sum(np.sum(u**2) for u in us)
      # scale-optimal: maximize sqrt(p/2)*||g||*sqrt(W2)/(U2+W2); h is 1-homog in u so ||g|| ∝ sqrt(U2): ratio max = sqrt(p/2)*||g||/sqrt(U2)/2
      r=np.sqrt(p/2)*np.linalg.norm(g)/np.sqrt(U2)/2
      best=max(best,r)
    print(p,n,"max sum|h^w^|/||w||^2 =",round(best,3)," paper bound p/4 =",p/4," p^n/4 =",p**n/4)
