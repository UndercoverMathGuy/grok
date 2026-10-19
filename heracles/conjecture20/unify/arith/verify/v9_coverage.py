# Referee: independent recomputation of the doubling-chain suprema (R9 step 3/4) from the certified base ratios,
# and coverage of every q in [17, 2*10^6] by (direct certificates) + (chains) + monotonicity.
import json, numpy as np, mpmath as mp
from fractions import Fraction
mp.mp.dps = 40
D = '/home/user/grok/heracles/conjecture20/unify/arith2/'
thr = mp.sqrt(mp.mpf(2)/3); r = mp.mpf(2)/3
cert = {}
for fn in ['compress_cert_uniform_s1.5.json', 'compress_cert_warm_s1.5.json', 'compress_cert_big_s1.5.json']:
    for q, e in json.load(open(D+fn)).items():
        bs = [mp.mpf(Fraction(x).numerator)/Fraction(x).denominator for x in e['bprime']]
        rho = (mp.fsum(b**r for b in bs))**(mp.mpf(3)/2)/mp.mpf(int(q))**(mp.mpf(3)/2)
        cert[int(q)] = min(cert.get(int(q), 9), rho*(1+mp.mpf('1e-30')))
chains = {}
for q0 in [84, 110, 144]:
    R = cert[q0]**r; qk = q0; sup = cert[q0]
    for k in range(1, 60):
        R = 2**(r-1)*(1+mp.mpf('1.01')/qk)**r*R + (mp.mpf('1.2')*qk)**r/(2*qk); qk *= 2
        rho = R**(mp.mpf(3)/2); cert[qk] = min(cert.get(qk, 9), rho); sup = max(sup, rho)
    chains[q0] = sup
    print(f"chain from q0={q0}: rho_q0={mp.nstr(cert[q0],7)}  sup over chain={mp.nstr(sup,7)}  last rho={mp.nstr(rho,6)}")
qs = sorted(cert); arr_q = np.array(qs, float); arr_r = np.array([float(cert[q]) for q in qs])
worst = 0; bad = []
for q in range(17, 2_000_001):
    i = np.searchsorted(arr_q, q)
    best = (arr_r[i:i+12]*(arr_q[i:i+12]/q)**1.5).min()
    if best >= float(thr): bad.append(q)
    worst = max(worst, best)
print(f"coverage q in [17, 2e6]: uncovered={bad[:10]} (count {len(bad)}), worst best-ratio={worst:.5f} vs threshold {float(thr):.5f}")
print("ladder ratios beyond 144:", sorted(set(round(qs[i+1]/qs[i], 4) for i in range(len(qs)-1) if qs[i] >= 144))[-3:])
