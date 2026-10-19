#!/bin/bash
# relaxation scan, one job per (prime, exponent), each under 15 minutes
cd /home/user/grok/heracles/conjecture20/unify/C_hunt
M="F,Sstar,AP8,sub4,sub6,sub8,sub10,sub12,sub14,sub16,sub18,sub20,sub22,sub24,sub26,sub28,sub30,sub32,sub34,sub36,sub38,sub42,sub46,sub48,sub50"
jobs=()
for p in 13 17 29 31 37 41 43 61 73 97 101 103 109 113 127 137 151 157 181 193 197; do
  jobs+=("0.24401693585629 1.7320508075688772 auto 1.5 $p b2")
  jobs+=("0.11388721241765587 1.2457309396155174 auto 1.2 $p b25")
done
printf '%s\n' "${jobs[@]}" | xargs -P 2 -L 1 bash -c 'OMP_NUM_THREADS=1 timeout 900 uv run --with numpy --with scipy --with mpmath python -u keff2.py $0 $1 $2 $3 $4 '"$M"' 2>&1 | grep --line-buffered -v UV_NATIVE > out2/keff2_$5_p$4.out'
