#!/bin/bash
# usage: p40-run.sh <arm> <dir holding libggml-cuda.so.0>   — depth-30K, 128-token decode, one repetition
set -u
A=$1; L=$2; D=/home/oscar/merlin-p40-cheap-20261007
export LD_LIBRARY_PATH=$L:/home/oscar/merlin-engine-stage-sm61:/home/oscar/merlin-p40-direct-bin
cd /home/oscar/merlin-p40-direct-bin
ldd ./llama-bench | grep -i "ggml-cuda\|ggml-base" > "$D/$A.ldd"
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,temperature.gpu,power.draw,utilization.gpu,power.limit --format=csv,noheader -l 10 > "$D/$A-gpu.csv" 2>/dev/null &
LOG=$!
./llama-bench -m /model/Ternary-Bonsai-2-27B-PQ2_0.gguf -d 30000 -n 128 -p 0 -b 4096 -ub 2048 -ctk q8_0 -ctv q8_0 -fa 1 -r 1 -t 8 -o json > "$D/$A.json" 2> "$D/$A.stderr"
RC=$?
kill $LOG 2>/dev/null
echo "arm=$A rc=$RC"
cat "$D/$A.ldd"
python3 - "$D/$A.json" <<'EOF'
import json, sys
for r in json.load(open(sys.argv[1])):
    print("row", r["n_prompt"], r["n_gen"], r["n_depth"], "s=%.3f" % (r["avg_ns"] / 1e9), "t/s=%.3f" % r["avg_ts"])
EOF
awk -F', ' 'NR>1{if($4>t)t=$4; if($5>p)p=$5; if($2>m)m=$2} END{print "gpu peak: mem_used", m, "temp", t, "power", p}' "$D/$A-gpu.csv"
