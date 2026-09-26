# Final certified graded curve: convex hull (in lam = kappa^{3/2}, mu = lam * a) of the certified points
# {(1, 0)} [a = 0, trivial]  U  two-order grid certificates (cert_grid_two.json)  U  {(sqrt3, 0.258 sqrt3)} [eta_two.py, kappa = 3^{1/3}].
# The valid set {(lam, mu): cost^{3/2} >= lam |G| - mu eta for all populations} is convex, so hull points are certified.
import json, numpy as np
two = json.load(open('cert_grid_two.json'))
pts = [(1.0, 0.0)] + [(r['kappa'] ** 1.5, r['kappa'] ** 1.5 * r['a']) for r in two if r['certified'] and r['a'] <= 0.25] + [(np.sqrt(3), 0.258 * np.sqrt(3))]
def lam_cert(a):
    best = 0
    for (l1, m1) in pts:
        for (l2, m2) in pts:
            # find theta with (m1 + t(m2 - m1)) = a (l1 + t(l2 - l1))
            den = (m2 - m1) - a * (l2 - l1)
            if abs(den) < 1e-15:
                if abs(m1 - a * l1) < 1e-12: best = max(best, l1, l2)
                continue
            t = (a * l1 - m1) / den
            if -1e-12 <= t <= 1 + 1e-12: best = max(best, l1 + t * (l2 - l1))
    return best
rows = []
print(" a      delta_cert   delta_lone   gap")
for a in np.round(np.arange(0.0, 0.2601, 0.01), 3):
    lam = min(lam_cert(a), np.sqrt(3)); d = np.sqrt(3) / lam; dl = max(1, np.sqrt(3) * (1 - np.sqrt(3) * a))
    rows.append(dict(a=float(a), delta_cert=d, delta_lone=dl)); print(f"{a:.2f}   {d:.4f}      {dl:.4f}      {d - dl:+.4f}")
json.dump(rows, open('final_curve.json', 'w'), indent=1)
