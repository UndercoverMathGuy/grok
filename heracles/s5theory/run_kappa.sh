#!/bin/bash
# sequential kappa numerics (local CPU)
cd "$(dirname "$0")"
F="uv run --with scipy python kappa_mm.py"
$F --nu 4 --q 2 --d 2 --starts 3 --R 16 20 --tag _check > out/kappa_nu4_q2.log 2>&1
$F --nu 3 --q 2.5 --d 2 3 --starts 3 --R 7 8 12 27 32 > out/kappa_nu3_q2.5.log 2>&1
$F --nu 5 --q 2 --d 2 --starts 3 --R 20 24 28 32 40 > out/kappa_nu5_q2.log 2>&1
$F --nu 3 --q 2 --d 4 --starts 3 --R 49 60 64 80 > out/kappa_nu3_q2_d4.log 2>&1
echo DONE > out/kappa_done.flag
