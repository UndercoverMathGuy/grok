#!/bin/bash
# usage: scripts/push.sh Heracles/Foo.lean [more files]   -- copies local files to the pod project
cd /Users/ruhaanrajadhyaksha/projects/grok/heracles/lean
for f in "$@"; do
  scp -q -i ~/.runpod/ssh/runpodctl-ssh-key -P 35137 "$f" root@213.173.111.104:/root/work/Heracles/"$f"
done
