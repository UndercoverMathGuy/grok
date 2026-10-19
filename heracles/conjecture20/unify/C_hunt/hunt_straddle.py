# HUNT (d): straddlers.  Seeds whose feature set is a UNION of two optimal supports (S* and a dilate d S*), class C,
# n = 2; does any such network reach W/D = 1 with more than K* features?  Usage: python hunt_straddle.py p b nrest seed
import sys, os, json, time
import numpy as np
import hlib as H

p = int(sys.argv[1]); bexp = float(sys.argv[2]); nrest = int(sys.argv[3]); seed = int(sys.argv[4])
n = 2; nu = 3; s = nu / bexp; r = 1 / s; F = (p - 1) // 2
rng = np.random.default_rng(seed)
G, y, bind = H.programme(p, s)
Sst = [l for l in range(1, F + 1) if y[l - 1] > 0]; D = nu ** (1 - s) * G


def red(l):
    l %= p
    return min(l, p - l)


dil = []
for d in range(2, F + 1):
    Sd = sorted(red(d * l) for l in Sst)
    if Sd != Sst and Sd not in [z[1] for z in dil]: dil.append((d, Sd))
print(f"# straddle p={p} b={bexp}: Gamma={G:.10f} S*={Sst} dilates={dil[:6]}", flush=True)
t0 = time.time(); recs = []
for it in range(nrest):
    d, Sd = dil[it % len(dil)]
    t = [0.5, 0.7, 0.85][(it // len(dil)) % 3]
    yd = np.zeros(F)
    for l in Sst: yd[red(d * l) - 1] += y[l - 1]
    # union network: IF blocks with amplitudes t^{1/r}-scaled (cost shares t and 1 - t)
    amp = {}
    for l in Sst: amp[l] = amp.get(l, 0) + t ** s * y[l - 1]
    for l in Sd: amp[l] = amp.get(l, 0) + (1 - t) ** s * yd[l - 1]
    sup = sorted(amp)
    freqs = [l for l in sup for _ in range(nu)]
    cc = H.CC(p, n, freqs, s)
    b = np.array([amp[l] / nu for l in freqs], complex) * np.exp(0.05 * rng.standard_normal(len(freqs)))
    cnt = {}; ph = np.zeros((len(freqs), n))
    for i, l in enumerate(freqs):
        k = cnt.get(l, 0); cnt[l] = k + 1; ph[i] = 2 * np.pi * k / nu
    ph += 0.05 * rng.standard_normal(ph.shape)
    x = cc.pack(b, ph)
    r0 = cc.gnorm(x) / D
    x = cc.lbfgs(x, taus=(1e-2, 3e-3, 1e-3, 3e-4), iters=1000)
    x, v = cc.slp(x, iters=150)
    fd = cc.feature_data(x); Qt = sum(fd[l]['Q'] for l in fd)
    sig = sorted(l for l in fd if fd[l]['Q'] / Qt > 1e-3)
    rec = dict(it=it, d=d, t=t, union=sup, start_ratio=r0, ratio=v / D, final_features=sig, count=len(sig), Kstar=len(Sst),
               final_is_dilate_of_Sstar=any(sorted(red(e * l) for l in Sst) == sig for e in range(1, F + 1)))
    recs.append(rec); print(json.dumps(rec, default=float), flush=True)
near = [z for z in recs if z['ratio'] > 1 - 1e-6]
print(f"# SUMMARY straddle p={p} b={bexp}: runs={len(recs)} best={max(z['ratio'] for z in recs):.10f} "
      f"near-opt counts={sorted(set(z['count'] for z in near))} K*={len(Sst)} "
      f"near-opt all single dilates={all(z['final_is_dilate_of_Sstar'] for z in near) if near else None} "
      f"best union-kept (count>K*) ratio={max([z['ratio'] for z in recs if z['count'] > len(Sst)] or [None])} time={time.time() - t0:.0f}s",
      flush=True)
