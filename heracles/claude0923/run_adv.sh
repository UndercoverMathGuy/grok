#!/bin/bash
# sequential adversarial AP-junk scan; one line per run appended to adv_scan.log
cd "$(dirname "$0")"
for p in 31 59; do
  if [ $p = 31 ]; then Ls="5 9 13"; ds="1 5"; else Ls="7 13"; ds="1 7"; fi
  for L in $Ls; do for d in $ds; do for s in 0.02 0.08 0.2 0.4; do
    r=$(uv run python adv_ap.py $p $L $d 2 1500 $s 2>&1 | tail -1)
    echo "p=$p L=$L d=$d sigma=$s | $r" >> adv_scan.log
  done; done; done
done
echo DONE >> adv_scan.log
