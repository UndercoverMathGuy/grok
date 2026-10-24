import numpy as np
s=1.5; r=1/s
def S(b,y):
    P=2*b*b+y*y; q=max(-2*b*b*y,P-1)
    nu=np.maximum(1+np.roots([1,0,-P,-q]).real,0)
    return np.mean(nu**r)**s
best=0
bs=np.linspace(0.005,1,200); ys=np.linspace(0,1,201); ds=np.linspace(0.005,1,200)
Sg=np.array([[S(b,y) for y in ys] for b in bs])
D,Y=np.meshgrid(ds,ys,indexing='ij')
mC=np.where(D>=Y,1.0,D*Y+np.sqrt(np.clip((1-D**2)*(1-Y**2),0,None)))
for i,b in enumerate(bs):
    R=(mC-Sg[i][None,:])/(2*b*b+D**2)
    k=np.unravel_index(R.argmax(),R.shape)
    if R[k]>best: best=R[k]; arg=(b,ys[k[1]],ds[k[0]])
print("grid sup K =",best,"at b,y,d=",np.round(arg,3))
# also with b=0 (B=C=0 case): sup over y,d of (mC-S(0,y))/d^2
R0=(mC-np.array([S(0,y) for y in ys])[None,:])/(D**2)
print("B=C=0 case sup K_D =",R0.max())
