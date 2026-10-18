# Coverage of all block lengths q >= 17 at s = 3/2 by certified dominators (+ monotonicity P_q' <= P_q for q' <= q).
# For each q in [17, 2000] report the best certified ratio  min_{certified q0 >= q} rho_{q0} (q0/q)^{3/2}, where the
# certified set is: individual certificates (JSON) and the doubling chains q0 2^k (q0 in {84,110,144}) with the
# interval-certified chain suprema from doubling_check.out (0.51247, 0.48579, 0.41890).
import json, numpy as np
thr = (2/3)**0.5
cert = {}
for fn in ['compress_cert_uniform_s1.5.json', 'compress_cert_warm_s1.5.json', 'compress_cert_big_s1.5.json']:
    for q, e in json.load(open(fn)).items():
        if e['ok']: cert[int(q)] = min(cert.get(int(q), 9), e['rho'])
chain = {84: 0.51247, 110: 0.48579, 144: 0.41890}
for q0, rho in chain.items():
    for k in range(1, 12): cert[q0*2**k] = min(cert.get(q0*2**k, 9), rho)
qs = sorted(cert)
bad = []; worst = 0
for q in range(17, 2001):
    best = min(cert[q0]*(q0/q)**1.5 for q0 in qs if q0 >= q)
    worst = max(worst, best)
    if best >= thr: bad.append(q)
print(f"certified block lengths: {qs[:30]} ...")
print(f"q in [17,2000]: uncovered = {bad};  worst best-ratio = {worst:.4f} (threshold {thr:.4f})")
print("beyond 2000 the three chains repeat with the same ratios (consecutive ladder ratios 110/84, 144/110, 168/144 <= 1.31 < 1/0.733)")
