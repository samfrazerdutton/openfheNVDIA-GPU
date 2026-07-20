#!/usr/bin/env bash
set -euo pipefail
REPO=~/openfheNVDIA-GPU
BUILD=$REPO/build
OUT=$REPO/results
REPS=7
mkdir -p "$OUT"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=$OUT/raw_$STAMP.log
MD=$OUT/RESULTS_$STAMP.md

echo "=== environment ===" | tee "$LOG"
nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,power.limit --format=csv | tee -a "$LOG"

nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,utilization.gpu,power.draw --format=csv -l 1 > "$OUT/telemetry_$STAMP.csv" &
TELE_PID=$!
trap 'kill $TELE_PID 2>/dev/null || true' EXIT

cd "$BUILD"

echo "=== warmup ===" | tee -a "$LOG"
./benchmark >/dev/null 2>&1 || true
./benchmark >/dev/null 2>&1 || true

run_reps () {
  local bin=$1
  echo "=== $bin x $REPS ===" | tee -a "$LOG"
  for i in $(seq 1 $REPS); do
    echo "--- rep $i ---" | tee -a "$LOG"
    ./"$bin" 2>&1 | tee -a "$LOG"
    sleep 3
  done
}

run_reps benchmark
run_reps bench_evalmult
run_reps test_e2e_ckks
run_reps bench_vs_cpu

median () { sort -n | awk '{a[NR]=$1} END {print (NR%2) ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2}'; }

{
  echo "# Benchmark results $STAMP"
  echo
  echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
  echo "Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)  CUDA: $(nvcc --version | grep release)"
  echo "Reps per benchmark: $REPS (median reported)"
  echo
  echo "| Metric | Median |"
  echo "|---|---|"
  M1=$(grep -oP '\[BENCH\] 16 towers x 32768 ring: \K[0-9.]+' "$LOG" | median)
  echo "| GPU pointwise, 16 towers x 32768 (ms/op) | $M1 |"
  M2=$(grep -oP '\[BENCH\] throughput: \K[0-9.]+' "$LOG" | median)
  echo "| GPU throughput (M coeff-mults/sec) | $M2 |"
  M3=$(grep -oP 'Mean Latency: \K[0-9.]+' "$LOG" | median)
  echo "| CPU OpenMP baseline latency (ms) | $M3 |"
  M4=$(grep -oP '\[3\] EvalMult: \K[0-9.]+' "$LOG" | median)
  echo "| CKKS EvalMult e2e (ms) | $M4 |"
  M5=$(grep -oP '\[2\] Encrypt: \K[0-9.]+' "$LOG" | median)
  echo "| CKKS Encrypt e2e (ms) | $M5 |"
  echo
  echo "Raw log: raw_$STAMP.log  Telemetry: telemetry_$STAMP.csv"
} > "$MD"

echo
echo "Done. Results:"
cat "$MD"
