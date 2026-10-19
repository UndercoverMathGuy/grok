#!/bin/bash
cd /home/user/grok/heracles/conjecture20/unify/C_hunt
until ! pgrep -f "[j]obs_a1" > /dev/null; do sleep 20; done
./run_jobs.sh jobs_rest.txt 3
echo ALLDONE > out/ALLDONE.flag
