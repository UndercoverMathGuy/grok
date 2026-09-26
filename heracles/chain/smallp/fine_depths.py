# Sharper certified depths for the supports that fail the isotropic route.
import json, sys
from depth_cert import certify
jobs = {'7': [([1, 2, 3], 36)], '11': [([1, 2, 3, 4], 16), ([1, 2, 3, 4, 5], 10)],
        '13': [([1, 2, 3, 4], 16), ([1, 2, 3, 5], 16), ([1, 2, 3, 6], 16), ([1, 2, 3, 4, 5], 10), ([1, 2, 3, 4, 5, 6], 8)]}
p = sys.argv[1]; out = []
for T, L in jobs[p]:
    a1, _ = certify(int(p), T, L, False); a2, _ = certify(int(p), T, L, True)
    row = dict(p=int(p), T=T, L=L, one_sided=a1, two_sided=a2); out.append(row); print(json.dumps(row), flush=True)
json.dump(out, open(f'fine_p{p}.json', 'w'), indent=1)
