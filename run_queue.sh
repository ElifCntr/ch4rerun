#!/usr/bin/env bash
# Waits for the current runs to finish, then works through the rest of the
# matrix. Skips any run that already has a summary.json, so it is safe to
# re-run and safe to start while something else is going.
cd ~/PycharmProjects/ch4rerun

# Three consecutive idle checks. The other loop's gap between seeds is
# seconds, so one idle poll does not mean the card is free; three minutes
# of quiet does.
idle=0
while false; do
  # pgrep -f matches this script too, since its own text contains the
  # string. Match the python process instead.
  if pgrep -f "python tools/train_arm.py" | grep -qv "^$$$"; then idle=0; else idle=$((idle+1)); fi
  sleep 60
done
echo "$(date -Iseconds) card free, starting queue"

for arm in r3d_18 tsm 2d_tframe_logitavg 2d_single; do
  for s in 1 2 3; do
    if [ -f "runs/${arm}_seed${s}/summary.json" ]; then
      echo "$(date -Iseconds) skip ${arm} seed ${s}, already done"
      continue
    fi
    echo "$(date -Iseconds) start ${arm} seed ${s}"
    python tools/train_arm.py --arm "$arm" --seed "$s" \
      > "runs/log_${arm}_${s}.txt" 2>&1
    echo "$(date -Iseconds) end ${arm} seed ${s} rc=$?"
  done
done
echo "$(date -Iseconds) queue complete"
