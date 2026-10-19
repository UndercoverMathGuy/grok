# Aggregate all hunt outputs into the tables of the report.
import glob, json, os, re
import numpy as np
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')


def recs(f):
    out = []
    for line in open(f):
        line = line.strip()
        if line.startswith('{'):
            try: out.append(json.loads(line))
            except Exception: pass
    return out


def summ(f):
    return [l.strip() for l in open(f) if 'SUMMARY' in l]


print("== (a) hunt_a: compensation, n=2, class C ==")
tot = 0
for f in sorted(glob.glob(f"{OUT}/hunt_a_*.out")):
    R = recs(f); tot += len(R)
    if not R: continue
    head = [l for l in open(f) if l.startswith('# a_')]
    best = max(z['ratio'] for z in R)
    near = [z for z in R if z['ratio'] > 1 - 1e-6]
    sig = [z['ratio'] for z in R if z.get('maxgain_sig', 0) > 1 + 1e-3] if 'maxgain_sig' in R[0] else None
    anyc = [z['ratio'] for z in R if z['maxgain'] > 1 + 1e-3]
    des = [z['designed'] for z in R if 'designed' in z]
    print(f"{os.path.basename(f)[7:-4]:28s} runs={len(R):3d} best={best:.10f} near-opt counts={sorted(set(z['count'] for z in near))} "
          f"K*_S={head[0].split('K*_S=')[1].split()[0] if head else '?'} best(any cheap)={max(anyc) if anyc else None} "
          f"best(sig cheap)={max(sig) if sig else None} best designed={max(des) if des else None}")
print("total hunt_a runs", tot)
for kind in ['fejer', 'safety', 'straddle']:
    print(f"== {kind} ==")
    for f in sorted(glob.glob(f"{OUT}/hunt_{kind}_*.out")):
        for l in summ(f): print("  ", l[:300])
        if kind == 'safety':
            R = recs(f)
            if R:
                byc = {}
                for z in R:
                    key = ('all' if len(z['cheap']) > 1 else 'one', z['c'])
                    byc[key] = max(byc.get(key, -9), z['best'])
                print("   ", os.path.basename(f), " ".join(f"{k[0]}:c={k[1]}:G={R[[ (('all' if len(z['cheap'])>1 else 'one'), z['c']) for z in R].index(k)]['gain']:.4f}->{v:.4f}" for k, v in sorted(byc.items())))
print("== (b) ==")
for f in sorted(glob.glob(f"{OUT}/hunt_b_*.out")):
    for l in summ(f): print("  ", l[:250])
for f in sorted(glob.glob(f"{OUT}/hunt_b2_*.out")):
    print("  ", os.path.basename(f))
    for z in recs(f):
        print(f"      profile={z['profile']:6s} gain={z['gain']:.4f} own_only={z['own_only']:.4f} with_foreign={z['with_foreign']:.4f} lift={z['lift']:.4f} foreign_out_energy={z.get('foreign_out_energy', 0):.3g}")
print("== (c) ==")
for f in sorted(glob.glob(f"{OUT}/hunt_c_*.out")):
    for l in summ(f): print("  ", l[:300])
print("== detection LP (multi-offset) ==")
for f in sorted(glob.glob(f"{OUT}/detect_lp_*.out")):
    for l in summ(f): print("  ", l[:200])
print("== jtest / jtest2 / ptest ==")
for f in sorted(glob.glob(f"{OUT}/jtest_*.out")):
    for l in summ(f): print("  ", l[:200])
for f in sorted(glob.glob(f"{OUT}/jtest2_*.out")) + sorted(glob.glob(f"{OUT}/ptest_*.out")):
    for z in recs(f): print("  ", json.dumps(z)[:260])
print("== VIOL files ==", glob.glob(f"{OUT}/VIOL_*"))
