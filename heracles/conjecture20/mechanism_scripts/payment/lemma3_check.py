# Exact-ish checks of the constants used in Lemma 3 (one-variable inequality), with mpmath at 50 digits.
import mpmath as mp
mp.mp.dps=50
x0=mp.sqrt(mp.mpf(2)/3); d0=1-x0
kap=lambda d: mp.mpf(1)/2*(1+d)*mp.log((3+d)/(3*(1+d)))+ (mp.mpf(1)/2*(1-d)*mp.log((3-d)/(3*(1-d))) if d<1 else 0)
print("x0=",x0,"d0=1-x0=",d0)
lowk=-(mp.mpf(2)/9*d0**2+d0**4/(12*(1-d0**2)))
print("kappa(d0)=",kap(d0)," lower bound used=",lowk, " ok:",kap(d0)>=lowk)
print("exp(lower bound)=",mp.e**lowk," RHS 2x0-2/3=",2*x0-mp.mpf(2)/3)
# series coefficients a_k
for k in range(1,6):
    ak=(1-(4*k+1)*mp.mpf(9)**(-k))/(2*k*(2*k-1)); print("a_%d="%k,ak, " 1/(2k(2k-1))=",mp.mpf(1)/(2*k*(2*k-1)))
# check series vs closed form
for d in [mp.mpf('0.1'),mp.mpf('0.5'),mp.mpf('0.9')]:
    ser=-mp.nsum(lambda k:(1-(4*k+1)*mp.mpf(9)**(-k))/(2*k*(2*k-1))*d**(2*k),[1,mp.inf])
    print("d=",d," closed",kap(d)," series",ser)
# g(lambda)=2 lam(2lam-1)(3/2)^(1-lam) <= 2 on [1/2,1]
print("max g:",max(2*l*(2*l-1)*mp.mpf(1.5)**(1-l) for l in mp.linspace(0.5,1,10001)))
print("max (45-a)/(9-a)^2 on [0,1]:",max((45-a)/(9-a)**2 for a in mp.linspace(0,1,1001)))
