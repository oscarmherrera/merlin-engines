#!/bin/bash
# usage: run-accept.sh <arm: mtp|base> <model.gguf>   -- fork llama-server on the Ryzen iGPU (Vulkan), port 18091
set -u
ARM=$1; MODEL=$2
S=/home/oscar/prism-vulkan-perf-20261008
LOG=/home/oscar/accept-$ARM.server.log
SPEC=""; [ "$ARM" = mtp ] && SPEC="--spec-type draft-mtp --spec-draft-n-max 3"
pkill -f "llama-server -m /model/mtp-src" 2>/dev/null; sleep 2
LD_LIBRARY_PATH=$S/build/bin VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.json \
  setsid nohup $S/build/bin/llama-server -m $MODEL --port 18091 --host 127.0.0.1 -ngl 999 -c 36864 -np 1 -fa on \
  -ctk q8_0 -ctv q8_0 -b 4096 -ub 2048 -t 16 --no-warmup $SPEC > $LOG 2>&1 < /dev/null &
for i in $(seq 1 120); do curl -s -m 3 http://127.0.0.1:18091/health 2>/dev/null | grep -q '"ok"' && break; sleep 5; done
curl -s -m 3 http://127.0.0.1:18091/health; echo
grep -iE "n_layer_nextn|nextn|spec|draft|error|fail" $LOG | grep -viE "token|key=" | head -8
python3 /home/oscar/mtp-accept.py $ARM http://127.0.0.1:18091 /home/oscar/accept-$ARM.jsonl 2>&1 | tail -8
grep -iE "draft|accept" $LOG | tail -6
pkill -f "llama-server -m /model/mtp-src"; sleep 3
echo "ACCEPT_ARM_DONE $ARM"
