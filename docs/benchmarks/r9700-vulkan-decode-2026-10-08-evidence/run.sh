#!/bin/bash
# usage: r9700-run.sh <arm> <depth> [extra llama-bench args]
# env overrides: CTK CTV (default q8_0), FA (default 1), NGEN (default 128), plus any GGML_VK_* passed through.
# Runs as root via sudo because the Prism libraries are root-only and oscar is not in the render group.
set -u
A=$1; DEPTH=$2; shift 2
D=/home/oscar/merlin-r9700-direct-20261008
export LD_LIBRARY_PATH=${LIBPRE:+$LIBPRE:}/opt/merlin/lib/prism-b10743-adfffbe:$D
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json
H=$(ls -d /sys/class/drm/card0/device/hwmon/hwmon* | head -1)
( while true; do echo "$(date -u +%FT%TZ), $(cat /sys/class/drm/card0/device/mem_info_vram_used), $(cat /sys/class/drm/card0/device/gpu_busy_percent), $(cat $H/temp1_input 2>/dev/null), $(cat $H/power1_average 2>/dev/null || cat $H/power1_input 2>/dev/null), $(grep '\*' /sys/class/drm/card0/device/pp_dpm_sclk | head -1), $(grep '\*' /sys/class/drm/card0/device/pp_dpm_mclk | head -1)"; sleep 5; done ) > "$D/$A-gpu.csv" 2>/dev/null &
LOG=$!
"$D/llama-bench" -m /model/Ternary-Bonsai-2-27B-PQ2_0.gguf -d "$DEPTH" -n "${NGEN:-128}" -p 0 -b 4096 -ub 2048 -ctk "${CTK:-q8_0}" -ctv "${CTV:-q8_0}" -fa "${FA:-1}" -r 1 -t 16 -o json "$@" > "$D/$A.json" 2> "$D/$A.stderr"
RC=$?
kill $LOG 2>/dev/null
echo "arm=$A depth=$DEPTH ctk=${CTK:-q8_0} fa=${FA:-1} rc=$RC env=$(env | grep -o '^GGML_VK_[A-Z_]*=[^ ]*' | tr '\n' ' ')"
python3 - "$D/$A.json" <<'EOF'
import json, sys
try:
    for r in json.load(open(sys.argv[1])):
        print("row", r["n_prompt"], r["n_gen"], r["n_depth"], "s=%.3f" % (r["avg_ns"] / 1e9), "t/s=%.3f" % r["avg_ts"])
except Exception as e:
    print("no json:", e)
EOF
awk -F', ' 'NR>1{if($2>m)m=$2; if($4>t)t=$4; if($5>p)p=$5} END{printf "gpu peak: vram %.0f MiB temp %.0f C power %.0f W\n", m/1048576, t/1000, p/1000000}' "$D/$A-gpu.csv"
