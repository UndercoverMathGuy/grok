#!/usr/bin/env bash
# heracles pod helper. Usage:
#   heracles/pod.sh up   <host> <port>            scp code, install deps
#   heracles/pod.sh run  <host> <port> <cohort>   launch jobs.py detached on the pod
#   heracles/pod.sh log  <host> <port> <cohort>   tail the cohort log
#   heracles/pod.sh score <host> <port> <cohort>  run score.py on the pod for every cell of the cohort
#   heracles/pod.sh scoreloop <host> <port> <cohort>  continuous full-instrument scoring loop (score_loop.sh)
#   heracles/pod.sh pull <host> <port> <cohort>   pull scores + logs (+ spectra/metrics/config, not weights); END ONLY
set -euo pipefail
KEY=~/.runpod/ssh/runpodctl-ssh-key
HERE=$(cd "$(dirname "$0")" && pwd)
cmd=$1; host=$2; port=$3; cohort=${4:-R}
SSH="ssh -i $KEY -p $port -o StrictHostKeyChecking=no root@$host"
case $cmd in
  up)
    $SSH "mkdir -p /workspace/heracles /workspace/heracles_runs"
    scp -i $KEY -P $port -o StrictHostKeyChecking=no "$HERE"/train.py "$HERE"/score.py "$HERE"/jobs.py "$HERE"/kstar.py "$HERE"/groups.py "$HERE"/score_loop.sh root@$host:/workspace/heracles/
    $SSH "pip install -q --break-system-packages safetensors 2>&1 | tail -1; python3 -c 'import torch;print(torch.__version__, torch.cuda.get_device_name(0))'" ;;
  run)
    $SSH "cd /workspace && (setsid nohup env RUNS_DIR=/workspace/heracles_runs python3 -u heracles/jobs.py --cohort $cohort > /workspace/heracles_${cohort}.log 2>&1 < /dev/null &) ; sleep 2; head -3 /workspace/heracles_${cohort}.log" ;;
  log)
    $SSH "tail -n 15 /workspace/heracles_${cohort}.log" ;;
  scoreloop)   # continuous full-instrument scoring next to training (see score_loop.sh)
    $SSH "cd /workspace && (setsid nohup bash heracles/score_loop.sh /workspace/heracles_${cohort}.log > /workspace/heracles_score_${cohort}.log 2>&1 < /dev/null &); sleep 1; echo started" ;;
  score)
    $SSH "cd /workspace && (setsid nohup sh -c 'for cell in heracles_runs/$cohort/*/; do n=\$(basename \$cell); python3 -u heracles/score.py --runs \$cell --out heracles_scores/$cohort/\$n.json --no-sides; done; echo SCORE DONE' > /workspace/heracles_score_${cohort}.log 2>&1 < /dev/null &)" ;;
  pull)
    mkdir -p "$HERE"/scores/$cohort "$HERE"/logs
    scp -i $KEY -P $port -o StrictHostKeyChecking=no -r root@$host:/workspace/heracles_scores/$cohort/. "$HERE"/scores/$cohort/ || true
    scp -i $KEY -P $port -o StrictHostKeyChecking=no root@$host:/workspace/heracles_${cohort}.log root@$host:/workspace/heracles_score_${cohort}.log "$HERE"/logs/ || true
    $SSH "cd /workspace && tar czf /workspace/${cohort}_light.tgz --exclude='*.safetensors' heracles_runs/$cohort" && scp -i $KEY -P $port -o StrictHostKeyChecking=no root@$host:/workspace/${cohort}_light.tgz "$HERE"/runs_${cohort}_light.tgz ;;
  *) echo "unknown cmd"; exit 1 ;;
esac
