#!/bin/bash
# Pull audit retrain runs from the pod into audit/retrain/, keep the queue manifests AND a fresh pod-side sha256 list,
# verify every file locally against both.
set -euo pipefail
H=47.47.180.47; P=14481; K=~/.runpod/ssh/runpodctl-ssh-key
O="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
L=/Users/ruhaanrajadhyaksha/projects/grok/heracles-emp/audit/retrain
R=/workspace/heracles-emp/audit/retrain_pod
mkdir -p $L/runs
ssh -i $K -p $P $O root@$H "cd $R/runs && find . -type f -print0 | xargs -0 sha256sum" > $L/pod_sha256.txt
scp -q -i $K -P $P $O "root@$H:$R/manifest_*.json" $L/
scp -q -i $K -P $P $O "root@$H:$R/RUNS.csv" $L/RUNS_pod.csv
scp -q -r -i $K -P $P $O "root@$H:$R/runs/*" $L/runs/
scp -q -i $K -P $P $O "root@$H:$R/logs/*" $L/ || true
cd $L/runs && shasum -a 256 -c ../pod_sha256.txt
cd $L && python3 - <<'PY'
import json,glob,hashlib,os
bad=0;n=0
for mf in glob.glob('manifest_*.json'):
    for rid,files in json.load(open(mf))['runs'].items():
        for fn,h in files.items():
            n+=1
            if hashlib.sha256(open(f'runs/{rid}/{fn}','rb').read()).hexdigest()!=h: bad+=1; print('BAD',rid,fn)
print('queue-manifest files checked',n,'bad',bad)
PY
