#!/bin/bash
cd /home/user/grok/heracles/conjecture20/unify/C_hunt
while pgrep -f "run_jobs.sh jobs_j.txt" > /dev/null; do sleep 15; done
./run_jobs.sh jobs_j3.txt 1
echo J3DONE
