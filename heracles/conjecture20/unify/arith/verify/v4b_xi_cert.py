# Referee: independently coded interval certificate for Xi_torus on s in [1,3/2] (40 s-boxes x 1000 t-boxes, 40 terms).
import mpmath as mp
iv = mp.iv; iv.dps = 30
# (b) independent interval certificate
def cert(ns=40, nt=1000, N=40):
    two3 = iv.mpf(2)/3; worst = mp.mpf(0)
    bc = [iv.mpf(mp.binomial(2*k, k))/iv.mpf(4)**k for k in range(N+1)]
    for i in range(ns):
        s1 = iv.mpf(1)+iv.mpf(i)/(2*ns); s2 = iv.mpf(1)+iv.mpf(i+1)/(2*ns)
        rmax = 1/s1
        # |binom(rmax, 2k)| as intervals
        co = []; bb = iv.mpf(1)
        for j in range(2*N):
            bb = bb*(rmax-j)/(j+1)
            if (j+1) % 2 == 0: co.append(abs(bb)*bc[(j+1)//2])
        c2 = 1-two3**(s2-1)
        for k in range(nt):
            t1 = iv.mpf(k)/nt; t2 = iv.mpf(k+1)/nt
            E = iv.mpf(1); tt = t1*t1; pw = tt
            for ck in co:
                E -= ck*pw; pw *= tt
            ub = (E/(1-c2*t2)**rmax).b
            if ub > worst: worst = ub
        # t >= 1 branch with r interval
        rI = iv.mpf([(1/s2).a, (1/s1).b])
        c0 = 2**rI*iv.gamma(rI+iv.mpf(1)/2)/(iv.sqrt(iv.pi)*iv.gamma(rI+1))
        tb = (c0*(iv.mpf(3)/2)**((s2-1)/s1)).b
        if tb > worst: worst = tb
    return worst

print('(b) sup Xi_torus over s in [1,1.5] <=', mp.nstr(cert(), 10))
