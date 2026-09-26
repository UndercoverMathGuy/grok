"""pax_tail.py -- why does N_.05 grow with p?  Analysis only, no training.
Reads archived REP metrics.json, rebuilds the clean energy spectrum en_k (share of g(D)'s energy at
frequency k), and tests the 'flat tail' model:  head of K frequencies holds fraction f of the energy,
the remaining 1-f is spread over the other n_f - K frequencies roughly uniformly.  Then
    N_.05  ~=  K + (n_f - K) * (0.95 - f) / (1 - f)
grows LINEARLY in n_f at fixed head quality f -- i.e. N_.05 is not a count, it is a tail statistic."""
import json, math
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent / "FINAL_RUNS" / "REP"
OUT = Path(__file__).resolve().parent / "FINAL_RUNS" / "PAX"; OUT.mkdir(parents=True, exist_ok=True)

def cells(act, init):
    for p in [59, 71, 89, 97, 113, 127, 157, 211]:
        tag = (f"n2_pow_p{p}_l2q2_lam0.0001_lrmax12.8_init{init}_m500_st40000" if act == "pow"
               else f"n2_prod_p{p}_l2q2_lam0.0001_lrmax12.8_init{init}_m500_st40000_clip1_wdw20000_esc0.8_lr212.8")
        d = R / tag
        if d.exists():
            yield p, d

rows = []
for act in ["pow", "prod"]:
    for p, d in cells(act, 4):
        recs = []
        for sd in sorted(d.glob("seed*")):
            m = json.loads((sd / "metrics.json").read_text())
            if m.get("acc", 0) < 0.999:
                continue
            amp = np.array(m["amp"]); en = np.sort(amp ** 2 / (amp ** 2).sum())[::-1]
            nf = len(en); cum = np.cumsum(en)
            N05 = int(np.searchsorted(cum, 0.95) + 1)
            K5 = int((en >= 0.05).sum()); K2 = int((en >= 0.02).sum())
            f5 = float(cum[K5 - 1])                       # energy in the >=5% head
            tail = en[K5:]
            # flat-tail prediction of N_.05 from (K5, f5, nf)
            pred = K5 + (nf - K5) * max(0.0, (0.95 - f5)) / max(1e-9, (1 - f5))
            # tail shape: ratio of tail mean to tail max, and effective count of the tail
            tpr = float(1 / (tail ** 2).sum() * (tail.sum()) ** 2) if len(tail) else 0.0
            recs.append((N05, K5, K2, f5, pred, tpr, float(en[0]), float(tail.sum()),
                         float(m["clean_frac"]), float(m["gamma"]), float(m["margin"])))
        if not recs:
            continue
        a = np.array(recs); med = np.median(a, 0)
        rows.append(dict(act=act, p=p, nf=(p - 1) // 2, n=len(recs), N05=med[0], K5=med[1], K2=med[2],
                         f_head=med[3], N05_pred=med[4], tail_PR=med[5], top=med[6], tail_en=med[7],
                         clean=med[8], gamma=med[9], margin=med[10]))

print(f"{'act':5s} {'p':>4s} {'n_f':>4s} {'n':>3s} {'K_5pct':>6s} {'f_head':>6s} {'N_.05':>6s} {'pred':>6s} "
      f"{'tailPR':>6s} {'tail/nf':>7s} {'top_en':>6s} {'clean':>5s} {'margin':>7s}")
for r in rows:
    print(f"{r['act']:5s} {r['p']:4d} {r['nf']:4d} {r['n']:3d} {r['K5']:6.1f} {r['f_head']:6.3f} "
          f"{r['N05']:6.1f} {r['N05_pred']:6.1f} {r['tail_PR']:6.1f} {r['tail_PR']/(r['nf']-r['K5']):7.3f} "
          f"{r['top']:6.3f} {r['clean']:5.3f} {r['margin']:7.3f}")
json.dump(rows, open(OUT / "tail.json", "w"), indent=1)
