# Referee: recompute every closed-form number in R2, R3, R6, R9 at high precision.
import mpmath as mp
mp.mp.dps = 30
def c_of(s): return 1-(mp.mpf(2)/3)**(s-1)
def tau(s): return (mp.mpf(3)/2)**(s-1)
s = mp.mpf(3)/2
c = c_of(s); tp = tau(s)
print("b=2: c12 =", mp.nstr(c, 12), " 1/c^2 =", mp.nstr(1/c**2, 12), " 2/c^2 =", mp.nstr(2/c**2, 12), " 1/(2c^2) =", mp.nstr(1/(2*c**2), 10))
print("1 + c tau' - tau' =", 1+c*tp-tp)
for p in [61, 67, 101, 211, 1009, 10007]:
    e = 1/(1+tp*max(c-mp.sqrt(mp.mpf(2)/p), 0))
    bp = 3/(1+mp.log(1-mp.sqrt(mp.mpf(2)/p))/mp.log(mp.mpf(2)/3))
    print(f"p={p}: e_thr(b=2) = {mp.nstr(e,6)}   b_p = {mp.nstr(bp,6)}")
print("limit (2/3)^(1/2) =", mp.nstr(mp.sqrt(mp.mpf(2)/3), 8), "; Note value 1/(1+c*sqrt3) =", mp.nstr(1/(1+c*mp.sqrt(3)), 6))
def isprime(n):
    if n < 2: return False
    i = 2
    while i*i <= n:
        if n % i == 0: return False
        i += 1
    return True
for b in ['2', '2.1', '2.25', '2.4', '2.5', '2.75', '2.9']:
    sb = 3/mp.mpf(b); cb = c_of(sb); thr = 2/cb**2
    P = max(q for q in range(3, int(thr)+1) if isprime(q) and q < thr)
    print(f"b={b}: s={mp.nstr(sb,6)} c={mp.nstr(cb,6)} 2/c^2={mp.nstr(thr,8)} largest prime below: {P}")
# alpha_p and the 1.1077 claim
def alpha(p, J=200000):
    C = (mp.mpf(8)/3)/mp.gamma(mp.mpf(1)/3)
    S = mp.fsum(1/(mp.mpf(j*p)*(j*p-1)**(mp.mpf(2)/3)) for j in range(1, J+1))
    S += (mp.mpf(3)/2)/(p*(J*p-1)**(mp.mpf(2)/3))
    return C*S
a61 = alpha(61)
print("alpha_61 <=", mp.nstr(a61, 8))
for XiT in [mp.mpf('1.068'), mp.mpf('1.0679940463')]:
    Xi = XiT + a61*(mp.mpf(3)/2)**(mp.mpf(1)/3)
    print(f"Xi_torus={XiT}: Xi_61 <= {mp.nstr(Xi,8)}  Xi^1.5 <= {mp.nstr(Xi**1.5, 8)};  with alpha=2.3e-3: {mp.nstr((XiT+mp.mpf('2.3e-3')*(mp.mpf(3)/2)**(mp.mpf(1)/3))**1.5, 8)}")
# c0 tail branch at s=3/2
r = 1/s
c0 = 2**r*mp.gamma(r+0.5)/(mp.sqrt(mp.pi)*mp.gamma(r+1))
print("c0(2/3) =", mp.nstr(c0, 8), " c0*tau'^r =", mp.nstr(c0*tp**r, 8))
# A<=0 branch: tau'^r * max_p-average of (cos)_+^r  < 1 ?
print("A<=0 branch: tau'^r * 2^{-r} (c0+alpha_61) =", mp.nstr(tp**r*2**(-r)*(c0+a61), 6))
# Gautschi constant
print("r/Gamma(1-r) at r=2/3:", mp.nstr(r/mp.gamma(1-r), 8))
# R9 numbers
print("max_x P_23:", mp.nstr(max(sum(1-mp.cos(j*x) for j in range(1,24)) for x in mp.linspace(2.5, mp.pi, 4001)), 8))
print("doubling: 1/cos(1/sqrt(84)) =", mp.nstr(1/mp.cos(1/mp.sqrt(84)), 8), " 1/sin(sqrt2/2) =", mp.nstr(1/mp.sin(mp.sqrt(2)/2), 8),
      " 2.4(1-1/(3*84)) =", mp.nstr(mp.mpf(2.4)*(1-mp.mpf(1)/252), 8), " y/(2 sin(y/2)) at y=sqrt2:", mp.nstr(mp.sqrt(2)/(2*mp.sin(mp.sqrt(2)/2))*2, 8))
thr = mp.sqrt(mp.mpf(2)/3)
for q0, rs in [(84, mp.mpf('0.51247')), (110, mp.mpf('0.48579')), (144, mp.mpf('0.41890'))]:
    print(f"chain {q0}: allowed q'/q >= {mp.nstr((rs/thr)**(mp.mpf(2)/3), 8)};  with ladder ratio 1.3095: ratio {mp.nstr(rs*mp.mpf('1.3095')**1.5, 6)}")
print("0.51247*(1/0.733)^1.5 =", mp.nstr(mp.mpf('0.51247')*(1/mp.mpf('0.733'))**1.5, 8), " vs thr", mp.nstr(thr, 8))
