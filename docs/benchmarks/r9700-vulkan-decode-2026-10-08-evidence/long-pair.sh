D=/home/oscar/merlin-r9700-direct-20261008; F=$D/fused3
for i in 1 2; do
  sudo -n env NGEN=512 $D/run.sh stock-d30k-n512-$i 30000 2>&1 | grep "^row"
  sudo -n env NGEN=512 LIBPRE=$F $D/run.sh fused3-d30k-n512-$i 30000 2>&1 | grep "^row"
done
echo LONGPAIR_DONE
