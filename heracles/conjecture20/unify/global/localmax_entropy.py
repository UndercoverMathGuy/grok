# Temperature analysis of the local maxima found by gsearch: for each saved network (fixed neurons), its ratio to the IF
# value as a function of s, its cost-share entropy H vs the IF entropy log nu + H(z*), and whether it would overtake IF.
import glob, json, math, re, numpy as np
from lib import *
from transitions_fast import table
from entropy_check import gam_s, H_V

def gIF(p, n, s):
    G = max(table(p, s).values()); return (n + 1) ** (1 - s) * Bp(p, n) * G

rows = []
for f in sorted(glob.glob("gs_p*_n*_b*_m*_r*_s*.npz")):
    m = re.match(r"gs_p(\d+)_n(\d+)_b([\d.]+)_m(\d+)_r(\d+)_s(\d+)\.npz", f)
    p, n, b = int(m.group(1)), int(m.group(2)), float(m.group(3))
    d = np.load(f); net = Net(d['U'], d['W'], b); nu = n + 1; s0 = nu / b
    if net.margin() <= 0: continue
    r0 = net.gamma() / gIF(p, n, s0)
    A, Q, a = net.cheapness(); feats = [j for j in range(len(A)) if A[j] > 1e-5 * A.max()]
    cheap = max(A[j] / a[j] for j in feats)
    if cheap < 1 + 1e-3 or r0 < 0.5: continue                    # non-IF local maxima only
    H, V = H_V(net, s0)
    G, y, _ = programme(p, s0, restarts=20); z = y ** (1 / s0); z = z[z > 1e-12]; z = z / z.sum()
    HIF = math.log(nu) + float(-(z * np.log(z)).sum())
    curve = [(round(s, 3), round(gam_s(net, s) / gIF(p, n, s), 4)) for s in np.linspace(1.0, nu / 2, 6)]
    rows.append((p, n, b, r0, len(feats), cheap, H, HIF, curve))
for (p, n, b, r0, cnt, cheap, H, HIF, curve) in sorted(rows, key=lambda t: -t[3])[:14]:
    print(f"p={p} n={n} b={b}: ratio {r0:.4f} count {cnt} max cheap {cheap:.3f} | H {H:.4f} vs IF {HIF:.4f} ({'lower' if H < HIF else 'higher'}) | ratio(s): {curve}")
