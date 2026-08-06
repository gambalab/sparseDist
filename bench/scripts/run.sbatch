#!/bin/bash
#SBATCH --job-name=sparsedist-bench
#SBATCH --output=bench/results/slurm/%A_%a.out
#SBATCH --error=bench/results/slurm/%A_%a.err
#
# EXCLUSIVE is not optional. Cells are timed to the millisecond and a shared
# node makes every number meaningless -- another job's memory bandwidth is
# indistinguishable from our own scaling behaviour.
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#
# %1 runs the array tasks ONE AT A TIME. Each task takes the whole node, so
# concurrent tasks would oversubscribe it and contaminate every measurement.
# The array exists to get restartability across walltime limits, not
# parallelism.
#SBATCH --array=0-7%1
#SBATCH --time=24:00:00
#
# Submit with:
#   mkdir -p bench/results/slurm
#   sbatch --export=ALL,BENCH_RUN_ID=$(date -u +%Y%m%dT%H%M%S)-$(git rev-parse --short HEAD) \
#          bench/scripts/run.sbatch
#
# BENCH_RUN_ID must be set ONCE and shared by every array task, or --resume
# looks for results under a run_id that no other task wrote and the design
# starts over.

set -euo pipefail

export BENCH_ROOT="${SLURM_SUBMIT_DIR}/bench"
export BENCH_LIB="${BENCH_LIB:-${SLURM_SUBMIT_DIR}/../sparseDist-bench-lib}"

# Thread pinning is applied per cell by thread_env(), which is what the scaling
# panel needs -- these are only the defaults for the driver process itself.
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "host        : $(hostname)"
echo "job / task  : ${SLURM_ARRAY_JOB_ID:-na} / ${SLURM_ARRAY_TASK_ID:-na}"
echo "cpus        : ${SLURM_CPUS_PER_TASK:-na}"
echo "run_id      : ${BENCH_RUN_ID:-unset}"
echo "bench root  : ${BENCH_ROOT}"

# Preflight gates the run. Exiting non-zero here is the whole reason it returns
# a status: a missing Bioconductor package should stop the job now, not at hour
# nine of twenty-four.
Rscript "${BENCH_ROOT}/scripts/preflight.R"

Rscript "${BENCH_ROOT}/scripts/run.R" \
  --run-id="${BENCH_RUN_ID}" \
  --stripe="${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT:-8}" \
  --cap=14400 \
  --resume

echo "task ${SLURM_ARRAY_TASK_ID:-na} complete"
