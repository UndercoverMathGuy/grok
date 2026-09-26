for p in 11 23 31 47 59 83 101; do
  for N in 1 2 3; do
    s=6; [ $N = 3 ] && s=2
    uv run --with torch python ptrend.py $p $N $s 3000 > trend_p${p}_N${N}.log 2>&1
  done
done
