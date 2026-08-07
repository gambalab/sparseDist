#!/bin/bash
#SBATCH --job-name=sparsedist-bench
#SBATCH --output=bench/results/slurm/%A_%a.out
#SBATCH --error=bench/results/slurm/%A_%a.err
#SBATCH --ntasks=1
#
# EXCLUSIVE AND WHOLE-NODE MEMORY ARE NOT TUNING CHOICES.
#
# --exclusive: cells are timed to the millisecond. A co-tenant job's memory
#   bandwidth is indistinguishable from our own scaling behaviour, so a shared
#   node makes the scaling panel meaningless -- not noisy, meaningless.
#
# --cpus-per-task=64: the thread ladder runs 1,2,4,8,16,32 with 64 as a
#   hyperthreading footnote. A smaller allocation makes the top of the ladder
#   oversubscribe; preflight reports it as CONFINED when num_procs < hw_threads.
#
# --mem=0 (all node memory): the frontier panel asks where each method stops
#   working. A dense 100k x 100k double result is 80 GB before input and
#   working copies, so a 120 GB cap would make the frontier a property of the
#   ALLOCATION rather than of the machine. If the cluster will not grant the
#   whole node, that is fine -- but then the manuscript must state the budget
#   the frontier was measured under.
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#
# %1 runs array tasks ONE AT A TIME. Each takes the whole node, so concurrent
# tasks would oversubscribe it. The array exists for restartability across
# walltime, not parallelism. This cluster has no walltime limit, so a single
# task would also work -- the array simply lets a crashed task resume.
#SBATCH --array=0-7%1
#
# SUBMIT:
#   mkdir -p bench/results/slurm
#   export BENCH_RUN_ID=$(date -u +%Y%m%dT%H%M%S)-$(git rev-parse --short HEAD)
#   echo $BENCH_RUN_ID | tee bench/results/CURRENT_RUN_ID
#   sbatch --export=ALL,BENCH_RUN_ID=$BENCH_RUN_ID,BENCH_PANEL=density,scaling,coercion \
#          bench/scripts/run.sbatch
#
# BENCH_RUN_ID must be set ONCE and shared by every array task and every later
# batch, or --resume looks under an id nobody wrote and the design starts over.

set -euo pipefail

export BENCH_ROOT="${SLURM_SUBMIT_DIR}/bench"
export BENCH_LIB="${BENCH_LIB:-${SLURM_SUBMIT_DIR}/../sparseDist-bench-lib-linux}"
export LC_ALL=C.UTF-8

# Which panels. Default to the cheap three; override at submit time.
PANELS="${BENCH_PANEL:-density,scaling,coercion}"
CAP="${BENCH_CAP:-14400}"

# Conda: the benchmark R is the env's R, not the system one.
if [ -n "${CONDA_EXE:-}" ]; then
  source "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"
  conda activate "${BENCH_CONDA_ENV:-sparsedist}"
fi

# Defaults for the driver process only; thread_env() re-pins per cell, which is
# what the scaling panel measures.
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "host        : $(hostname)"
echo "job / task  : ${SLURM_ARRAY_JOB_ID:-na} / ${SLURM_ARRAY_TASK_ID:-na}"
echo "cpus        : ${SLURM_CPUS_PER_TASK:-na}"
echo "mem         : ${SLURM_MEM_PER_NODE:-na}"
echo "run_id      : ${BENCH_RUN_ID:-unset}"
echo "panels      : ${PANELS}"
echo "R           : $(which Rscript)"

# Preflight gates the run: a missing package should stop this now, not at hour
# nine. It exits non-zero on failure, and set -e turns that into a dead task.
Rscript "${BENCH_ROOT}/scripts/preflight.R"

Rscript "${BENCH_ROOT}/scripts/run.R" \
  --run-id="${BENCH_RUN_ID}" \
  --panel="${PANELS}" \
  --stripe="${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1}" \
  --cap="${CAP}" \
  --resume

echo "task ${SLURM_ARRAY_TASK_ID:-na} complete"
