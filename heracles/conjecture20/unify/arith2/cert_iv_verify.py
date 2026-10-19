# Interval-arithmetic re-verification (mpmath.iv, outward rounding) of every compression certificate in the JSON files:
#   g(x) = sum_l b'_l (1 - cos l x) - P_q(x) >= 0 on [0, pi],   P_q(x) = q + 1/2 - sin((q+1/2)x)/(2 sin(x/2)),
#   b'_l rational (the stored decimal strings, denominators 10^9).
# [0, x_s]:  x_s = lower endpoint of the enclosure of sqrt(12 (S2' - S2)/S4')  (Taylor: 1 - cos t in [t^2/2 - t^4/24, t^2/2]);
# [x_s, pi]: cells [x_i, x_{i+1}] with exact binary endpoints; g >= min(g(x_i), g(x_{i+1})) - M2 h^2/8, M2 = S2' + S2 >= sup|g''|,
#            every quantity enclosed by intervals; the last cell ends at the upper endpoint of the enclosure of pi.
# Also recomputes rho = (sum b'^r)^s / q^s as an interval upper bound.
import json, sys
import mpmath as mp
iv = mp.iv; iv.dps = 30

def verify(q, s, freqs, bstr):
    r = iv.mpf(1)/iv.mpf(s) if s != 1.5 else iv.mpf(2)/3
    sI = iv.mpf(3)/2 if s == 1.5 else iv.mpf(s)
    b = [iv.mpf(mp.mpf(x)) for x in bstr]          # decimal strings with <= 9 decimals: exact in 30-digit intervals up to rounding enclosure
    S2p = sum(bl*l*l for bl, l in zip(b, freqs)); S4p = sum(bl*l**4 for bl, l in zip(b, freqs))
    S2 = iv.mpf(q*(q+1)*(2*q+1))/6
    if (S2p - S2).a <= 0: return False, 'S2', None
    xsI = iv.sqrt(12*(S2p - S2)/S4p); xs = mp.mpf(xsI.a)
    M2 = (S2p + S2).b
    half = iv.mpf(1)/2
    def g(x):
        X = iv.mpf(x)
        return sum(bl*(1 - iv.cos(l*X)) for bl, l in zip(b, freqs)) - (q + half - iv.sin((q+half)*X)/(2*iv.sin(X/2)))
    PI = mp.mpf(iv.pi.b)
    x = xs; gx = g(x).a; worst = mp.inf; n = 0
    while x < PI:
        h = min(mp.mpf('0.02')*x, PI/600)
        while True:
            x2 = min(x + h, PI); g2 = g(x2).a
            lb = (iv.mpf(min(gx, g2)) - iv.mpf(M2)*iv.mpf(x2 - x)**2/8).a
            if lb >= 0 or h < mp.mpf('1e-12'): break
            h = h/2
        if lb < 0: return False, f'cell at x={mp.nstr(x, 8)}', None
        worst = min(worst, lb); x, gx = x2, g2; n += 1
    rho = (sum(bl**r for bl in b)**sI / iv.mpf(q)**sI).b
    return True, f'cells={n} min lb={mp.nstr(worst, 3)}', rho

if __name__ == '__main__':
    jobs = [('compress_cert_uniform_s1.5.json', 1.5), ('compress_cert_warm_s1.5.json', 1.5), ('compress_cert_big_s1.5.json', 1.5),
            ('compress_cert_b225_s1.3333333333333333.json', 4/3)]
    for fn, s in jobs:
        try: d = json.load(open(fn))
        except FileNotFoundError: continue
        for q, e in sorted(d.items(), key=lambda kv: int(kv[0])):
            ok, info, rho = verify(int(q), s, e['freqs'], e['bprime'])
            thr = (2/3)**(s-1)
            print(f"{fn} q={q}: interval-verified={ok} ({info}); rho <= {mp.nstr(rho, 6) if rho is not None else '-'} "
                  f"(threshold {thr:.4f}, {'below' if (rho is not None and rho < thr) else 'NOT below'})", flush=True)
