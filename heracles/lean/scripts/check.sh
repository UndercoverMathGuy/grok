#!/bin/bash
# usage: scripts/check.sh Heracles/Foo.lean  -- push and elaborate one file on the pod (timeout 9 min)
cd /Users/ruhaanrajadhyaksha/projects/grok/heracles/lean
./scripts/push.sh "$1"
ssh -i ~/.runpod/ssh/runpodctl-ssh-key -p 35137 root@213.173.111.104 "cd /root/work/Heracles && source ~/.elan/env && timeout 540 lake env lean $1 2>&1 | head -${2:-60}"
