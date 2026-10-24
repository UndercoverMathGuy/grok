#!/bin/sh
R="uv run --with numpy --with scipy --with mpmath python"
export OMP_NUM_THREADS=1
for f in cubic_check psi_check closure_check closure_unified closure_multiset split_leg_check pw_certify skew_adv fre_stored fre_hunt; do
  timeout 900 $R $f.py > $f.out 2>&1
done
timeout 900 $R pw_cex.py > pw_cex.out 2>&1
timeout 900 $R verify_net.py pwpos_p43_n2_s1.5.npy pwpos_p59_n2_s1.5.npy > verify_net.out 2>&1
timeout 900 $R pw_legendre.py > pw_legendre.out 2>&1
timeout 900 $R pwe_search.py > pwe_search.out 2>&1
( for a in "3 2 1.5 3" "3 3 2.0 4" "3 4 2.5 5" "5 2 1.5 3" "5 3 2.0 4" "5 4 2.5 5" "7 2 1.5 4" "7 3 2.0 5" "11 2 1.5 4"; do timeout 900 $R sf_pw.py $a 8 re; done ) > sf_pw.out 2>&1
echo done > run_all.done
