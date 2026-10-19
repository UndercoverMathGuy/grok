# Aggregate the resumed hunt (out2/) into the report tables.
import glob, json, os, math, collections
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out2')


def recs(f):
    out = []
    for line in open(f):
        line = line.strip()
        if line.startswith('{'):
            try: out.append(json.loads(line))
            except Exception: pass
    return out


def head(f):
    for line in open(f):
        if line.startswith('# ') and 'SUMMARY' not in line: return line.strip()
    return ''


print("== (a) hunt_a2: class C, n = 2, exact programme, full-active-set polish ==")
tot = 0; allbest = 0; viol = []
for f in sorted(glob.glob(f"{OUT}/hunt_a2_*.out")):
    R = recs(f); tot += len(R)
    if not R: print(os.path.basename(f), 'no records (timeout before first restart)'); continue
    h = head(f)
    Kst = h.split('K*_S=')[1].split()[0] if 'K*_S=' in h else '?'
    best = max(z['ratio'] for z in R); allbest = max(allbest, best)
    bestT = max([z.get('ratio_T', -9) for z in R])
    des = [z['designed'] for z in R if 'designed' in z]
    bydes = collections.defaultdict(list)
    for z in R:
        if 'designed' in z:
            key = z['fam'] + (f"(c={z['c']})" if 'c' in z else '')
            bydes[key].append(z['designed'])
    near = [z for z in R if z['ratio'] > 1 - 1e-6]
    cheap = [z['ratio'] for z in R if z['maxgain_sig'] > 1 + 1e-3]
    print(f"{os.path.basename(f)[8:-4]:24s} runs={len(R):2d} best ratio_S={best:.10f} best ratio_T={bestT:.10f} K*_S={Kst} "
          f"near-opt counts={sorted(set(z['count'] for z in near))} best final w/ sig. cheap feature={max(cheap) if cheap else None} "
          f"designed(max by family)={ {k: round(max(v), 4) for k, v in bydes.items()} }")
    viol += [z for z in R if z.get('ratio_T', 0) > 1 + 1e-9]
print(f"total hunt_a2 restarts={tot}; overall best ratio={allbest:.12f}; violations={len(viol)}")
print("== dictionary LP ==")
for f in sorted(glob.glob(f"{OUT}/dict_lp_*.out")):
    R = recs(f)
    if not R: continue
    print(f"{os.path.basename(f)[8:-4]:24s} runs={len(R):2d} best LP={max(z['lp_ratio'] for z in R):.6f} best polished={max(z['polished'] for z in R):.12f} "
          f"max nonpure in LP solution={max(z['nonpure'] for z in R)} final feature sets={sorted(set(tuple(z['feats']) for z in R))}")
print("== (c) hunt_c2 ==")
for f in sorted(glob.glob(f"{OUT}/hunt_c2_*.out")):
    R = recs(f)
    for l in open(f):
        if 'SUMMARY' in l: print('  ', l.strip()[:330])
    if not any('SUMMARY' in l for l in open(f)) and R:
        p1 = [z for z in R if z['part'] == 1]; p2 = [z for z in R if z['part'] == 2]; p3 = [z for z in R if z['part'] == 3]
        print('  ', os.path.basename(f), 'partial:', 'P1 frozen best=%s released best=%s' % (max([z['frozen']['ratio'] for z in p1] or [None]), max([z['free']['ratio'] for z in p1] or [None])),
              'P2 best=%s' % max([z['ratio'] for z in p2] or [None]), 'P3 best=%s' % max([z['ratio'] for z in p3] or [None]))
print("== (d) straddlers ==")
for f in sorted(glob.glob(f"{OUT}/hunt_straddle2_*.out")):
    for l in open(f):
        if 'SUMMARY' in l: print('  ', l.strip()[:330])
print("== joint detection constant (avg pair, CCP-LP) vs sqrt((p-1)/(p+1)) ==")
for f in sorted(glob.glob(f"{OUT}/detl2_*kseed.out")):
    for z in recs(f):
        p = z['p']; print(f"   p={p:3d} n={z['n']} rho={z['rho2']:.8f} formula={math.sqrt((p-1)/(p+1)):.8f} starts={z['starts']}")
print("== L2 payment constants (ptest2) ==")
for f in sorted(glob.glob(f"{OUT}/ptest2_*.out")):
    R = recs(f)
    if R and 'theta2' in R[0]: print('  ', json.dumps({k: R[0][k] for k in R[0] if k in ('n', 's', 'Lambda', 'theta2', 'at_G', 'at_eta', 'm', 'masses')}))
print("== VIOL files ==", sorted(glob.glob(f"{OUT}/VIOL_*")))
