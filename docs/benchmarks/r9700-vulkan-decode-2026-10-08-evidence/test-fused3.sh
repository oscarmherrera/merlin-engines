#!/bin/bash
# fused3 = fusion 1 (sign-fused FWHT) + delta-net parity (raw gates, rows mode, fused cache write)
D=/home/oscar/merlin-r9700-direct-20261008
F=$D/fused3
ICD=/usr/share/vulkan/icd.d/radeon_icd.json
cd $D && rm -rf fused3 && tar xzf fused3.tgz && cd fused3 || exit 1
for f in libggml-vulkan libggml-base libggml-cpu libggml; do ln -sfn $f.so.0.21.0 $f.so.0; done
ln -sfn libllama.so.0.2.0 libllama.so.0; ln -sfn libllama-common.so.0.2.0 libllama-common.so.0
chmod +x test-backend-ops llama-simple
echo "## correctness GATED_DELTA_NET:"
sudo -n env LD_LIBRARY_PATH=$F VK_ICD_FILENAMES=$ICD ./test-backend-ops test -b Vulkan0 -o GATED_DELTA_NET 2>&1 | sed "s/\x1b\[[0-9;]*m//g" | grep -E "tests passed|FAIL|not supported" | sort | uniq -c
echo "## correctness MUL_MAT_HADAMARD:"
sudo -n env LD_LIBRARY_PATH=$F VK_ICD_FILENAMES=$ICD ./test-backend-ops test -b Vulkan0 -o MUL_MAT_HADAMARD 2>&1 | sed "s/\x1b\[[0-9;]*m//g" | grep -E "tests passed|FAIL"
echo "## text equality (greedy, 48 tokens):"
P="The three laws of thermodynamics are"
sudo -n env LD_LIBRARY_PATH=$F VK_ICD_FILENAMES=$ICD $F/llama-simple -m /model/Ternary-Bonsai-2-27B-PQ2_0.gguf -n 48 "$P" 2>/dev/null > $D/text-fused3.txt
echo "fused3: $(wc -c < $D/text-fused3.txt) bytes  stock: $(wc -c < $D/text-stock.txt) bytes"
cmp $D/text-stock.txt $D/text-fused3.txt && echo TEXT_IDENTICAL || echo TEXT_DIFFERS
echo "## decode pair:"
sudo -n env LIBPRE=$F $D/run.sh fused3-d30k 30000 2>&1 | grep "^row"
sudo -n $D/run.sh stock-d30k-6 30000 2>&1 | grep "^row"
sudo -n env LIBPRE=$F $D/run.sh fused3-d30k-2 30000 2>&1 | grep "^row"
echo "## profiler:"
sudo -n rm -f $D/perf-fused3.stderr $D/perf-fused3.json
sudo -n env NGEN=32 LIBPRE=$F GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_FREQUENCY=1 $D/run.sh perf-fused3 30000 2>&1 | grep "^row"
sudo -n chown oscar:oscar $D/perf-fused3.stderr $D/text-*.txt $D/fused3-d30k*.stderr 2>/dev/null
grep -c "Vulkan Timings" $D/perf-fused3.stderr
grep -c "GDN_CACHE" $D/perf-fused3.stderr
grep -iE "gated_delta|GDN" $D/perf-fused3.stderr | tail -3
echo "## fused3 stderr errors:"; grep -iE "error|assert|abort" $D/fused3-d30k.stderr | head -5
echo DONE_FUSED3
