import json, sys
from pw_adv import adv
p = int(sys.argv[1]); d = json.load(open(f'/home/user/grok/heracles/conjecture20/mechanism_scripts/relations3/large_{p}.json'))
runs = sorted(d['runs'], key=lambda x: -x['Gamma'])
seen = set()
for run in runs:
    if run['K'] in seen: continue
    seen.add(run['K'])
    val, info = adv(p, run['S'], run['y'], restarts=6, iters=200)
    print(f"p={p} K={run['K']} Gamma={run['Gamma']:.5f}: max_adv min_w C(w)/sum a^r = {val:.5f}  max t={max(info['t']):.3f}", flush=True)
