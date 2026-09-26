while pgrep -f "ptrend.py 23 2" >/dev/null; do sleep 5; done
uv run --with torch python ptrend.py 23 3 2 3000 > trend_p23_N3.log 2>&1
for p in 31 47 59; do for N in 1 2 3; do s=6; [ $N = 3 ] && s=2; [ $p = 59 ] && s=3; [ $p = 59 ] && [ $N = 3 ] && s=1
  uv run --with torch python ptrend.py $p $N $s 3000 > trend_p${p}_N${N}.log 2>&1; done; done
for p in 83 101; do for N in 2 1 3; do s=2; [ $N = 3 ] && s=1
  uv run --with torch python ptrend.py $p $N $s 1500 > trend_p${p}_N${N}.log 2>&1; done; done
