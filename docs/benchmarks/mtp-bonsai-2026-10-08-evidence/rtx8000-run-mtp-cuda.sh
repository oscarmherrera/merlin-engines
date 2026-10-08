#!/bin/bash
# usage: run-mtp-cuda.sh <arm: mtp|base> <depths> [max_tokens]   -- direct merlin-endpoint on :8091, CUDA Prism libs + patched libllama
set -u
ARM=$1; DEPTHS=$2; MAXTOK=${3:-512}
D=/tmp/merlin-rtx8000-mtp
L=$D/libs
HEAD=$D/mtp-Qwen3.8-27B-Q8_0.gguf
MTPFLAGS=""; case $ARM in mtp*) MTPFLAGS="--mtp-model $HEAD --mtp-draft-max ${DRAFT_MAX:-3}";; esac
LOG=$D/mtp-$ARM.endpoint.log
sudo -n pkill -f "merlin-endpoint-mtp" 2>/dev/null; sleep 2
sudo -n env ${EXTRA_ENV:-} YZMA_LIB=$L LD_LIBRARY_PATH=$L \
  setsid nohup $D/merlin-endpoint-mtp --model /model/Ternary-Bonsai-2-27B-PQ2_0.gguf --alias bonsai-mtp-test --addr :8091 \
  --ctx-size 65536 --parallel 1 --n-gpu-layers 999 --cache-type-k q8_0 --cache-type-v q8_0 --batch-size 4096 --ubatch-size 2048 \
  --lib $L $MTPFLAGS > $LOG 2>&1 < /dev/null &
for i in $(seq 1 120); do curl -s -m 3 http://127.0.0.1:8091/state >/dev/null 2>&1 && break; sleep 5; done
curl -s -m 5 http://127.0.0.1:8091/state | python3 -c 'import json,sys; d=json.load(sys.stdin); print("state:", {k:d.get(k) for k in ("alias","slots","context_size","endpoint_build")})'
sudo -n grep -E "MTP head|Failed|Engine ready" $LOG | sed -E "s/(token|key)[^ ]*/\1=REDACTED/gi" | cut -c1-200 | head -4
sudo -n chmod a+r $LOG
python3 $D/measure-runtime.py --label $ARM --base http://127.0.0.1:8091 --depths $DEPTHS --max-tokens $MAXTOK --out $D/mtp-$ARM.jsonl --log-file $LOG 2>&1 | tail -6
echo "--- natural prompts ($ARM):"
python3 $D/prose-mtp.py http://127.0.0.1:8091 $ARM $LOG 2>&1 | tail -8
sudo -n pkill -f "merlin-endpoint-mtp"; sleep 3
echo "ARM_DONE $ARM"
