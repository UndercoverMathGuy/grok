"""Close a pod row in COMPUTE.md: python experiments/common/ledger_close.py POD_ID 'YYYY-MM-DD HH:MM' NOTE"""
import sys, datetime
pid, t_del, note = sys.argv[1], sys.argv[2], sys.argv[3]
p = "COMPUTE.md"; s = open(p).read()
i = s.index("| " + pid + " |"); j = s.index("\n", i)
cells = s[i:j].split("|")
rate = float(cells[4]); t0 = datetime.datetime.strptime(cells[5].strip(), "%Y-%m-%d %H:%M"); t1 = datetime.datetime.strptime(t_del, "%Y-%m-%d %H:%M")
hrs = (t1 - t0).total_seconds() / 3600
cells[6] = f" {t_del} "; cells[7] = f" {hrs:.2f} "; cells[8] = f" ${hrs * rate:.2f} "; cells[9] = cells[9].rstrip() + f"; {note} "
s = s[:i] + "|".join(cells) + s[j:]
open(p, "w").write(s)
print(pid, f"{hrs:.2f} h", f"${hrs * rate:.2f}")
