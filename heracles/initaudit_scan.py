"""initaudit_scan.py -- summarise the init-scale control cells from saved metrics.json (no retraining)."""
import json, statistics as st
from pathlib import Path
ROOT = Path(__file__).resolve().parent
REP = ROOT / "FINAL_RUNS" / "REP"

cells = {}
for f in sorted(REP.glob("n2_pow_*_l2q2_*/seed*/metrics.json")):
    r = json.loads(f.read_text())
    if r["q"] != 2.0 or r["lam"] != 1e-4 or "_from" in f.parent.parent.name: continue
    cells.setdefault((r["p"], r["init_scale"]), []).append(r)

def med(xs): return st.median(xs) if xs else float("nan")
print(f"{'p':>4} {'init':>5} {'n':>3} {'fit':>4} {'accmin':>7} {'K2':>10} {'K5':>10} {'N05':>10} {'clean':>6} {'norm2':>9} {'gamma':>10} {'margin':>8} {'ce':>8} {'esc':>7}")
for k in sorted(cells):
    rs = sorted(cells[k], key=lambda r: r["seed"]); fit=[r for r in rs if r["acc"]>=0.999]
    if not fit:
        print(f"{k[0]:>4} {k[1]:>5g} {len(rs):>3} {0:>4} {min(r['acc'] for r in rs):>7.3f}  --- collapsed ---")
        continue
    # escape step: first log step with acc>=0.99
    esc=[]
    for r in fit:
        e=[L["step"] for L in r["log"] if L["acc"]>=0.99]
        esc.append(e[0] if e else None)
    print(f"{k[0]:>4} {k[1]:>5g} {len(rs):>3} {len(fit):>4} {min(r['acc'] for r in rs):>7.3f} "
          f"{med([r['K_2pct'] for r in fit]):>4} [{min(r['K_2pct'] for r in fit)}-{max(r['K_2pct'] for r in fit)}] "
          f"{med([r['K_5pct'] for r in fit]):>4} [{min(r['K_5pct'] for r in fit)}-{max(r['K_5pct'] for r in fit)}] "
          f"{med([r['N_05'] for r in fit]):>4} [{min(r['N_05'] for r in fit)}-{max(r['N_05'] for r in fit)}] "
          f"{med([r['clean_frac'] for r in fit]):>6.3f} {med([r['norm2'] for r in fit]):>9.1f} "
          f"{med([r['gamma'] for r in fit]):>10.3e} {med([r['margin'] for r in fit]):>8.3f} "
          f"{med([r['final_ce'] for r in fit]):>8.4f} {str(med([e for e in esc if e])):>7}")
