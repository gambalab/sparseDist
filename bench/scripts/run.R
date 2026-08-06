#!/usr/bin/env Rscript
## ---------------------------------------------------------------------------
## run.R -- execute the benchmark design.
##
##   Rscript bench/scripts/run.R --dry-run
##   Rscript bench/scripts/run.R --panel=density,scaling
##   Rscript bench/scripts/run.R --stripe=3/8 --resume
##
## OPTIONS
##   --dry-run       build and summarise the design, run nothing
##   --panel=a,b     restrict to named panels
##   --stripe=i/n    run stripe i of n (SLURM array tasks)
##   --resume        skip cells already completed for this run_id
##   --cap=SECONDS   wall-clock cap per cell (default 14400 = 4h)
##   --run-id=ID     reuse an existing run_id (required for --resume)
##   --limit=N       run only the first N cells of the (shuffled) stripe
##   --allow-no-openmp
##                   proceed on a build with no OpenMP. Every cell then records
##                   status "dry_run" with timing and memory fields blank, so
##                   the pipeline is exercised end to end and no unusable
##                   number can reach a plot. This is the Mac rehearsal.
##
## RESUME IS NOT OPTIONAL AT THIS SCALE. Over a thousand cells at up to four
## hours each cannot fit one job's walltime, so the design is meant to be run
## as a restartable sequence: each invocation completes what it can and the
## next picks up the remainder. A cell counts as done when its result file
## exists, validates, and carries this run_id.
## ---------------------------------------------------------------------------

script_path <- function() {
  ca  <- commandArgs(trailingOnly = FALSE)
  hit <- grep("^--file=", ca, value = TRUE)
  if (length(hit)) normalizePath(sub("^--file=", "", hit[1]), mustWork = FALSE)
  else NA_character_
}
sp   <- script_path()
root <- Sys.getenv("BENCH_ROOT", unset = if (is.na(sp)) getwd()
                                        else dirname(dirname(sp)))
source(file.path(root, "R", "load-harness.R"))
load_harness(root, require_all = TRUE)
source(file.path(root, "R", "20-design.R"))

args <- commandArgs(trailingOnly = TRUE)
opt <- function(name, default = NULL) {
  hit <- grep(paste0("^--", name, "="), args, value = TRUE)
  if (!length(hit)) return(default)
  sub(paste0("^--", name, "="), "", hit[1])
}
dry_run   <- "--dry-run" %in% args
resume    <- "--resume"  %in% args
allow_serial <- "--allow-no-openmp" %in% args
limit     <- suppressWarnings(as.integer(opt("limit", NA)))
cap_sec <- as.numeric(opt("cap", "14400"))
panels  <- strsplit(opt("panel", paste(BENCH_PANELS, collapse = ",")), ",")[[1]]

## run_id ties every result row to one invocation of the design. It must be
## STABLE across restarts, or --resume finds nothing and the whole design runs
## again from scratch.
run_id <- opt("run-id", Sys.getenv("BENCH_RUN_ID", unset = ""))
if (!nzchar(run_id)) {
  sha <- tryCatch(system2("git", c("-C", shQuote(dirname(root)), "rev-parse",
                                   "--short", "HEAD"),
                          stdout = TRUE, stderr = FALSE),
                  error = function(e) "nogit")
  if (!length(sha) || !nzchar(sha[1])) sha <- "nogit"
  run_id <- paste0(format(Sys.time(), "%Y%m%dT%H%M%S", tz = "UTC"), "-", sha[1])
}

out_dir <- file.path(root, "results", "cells", run_id)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cat("run_id : ", run_id, "\n", sep = "")
cat("panels : ", paste(panels, collapse = ", "), "\n", sep = "")
cat("cap    : ", cap_sec, " s per cell\n", sep = "")
cat("out    : ", out_dir, "\n", sep = "")

## --- design -----------------------------------------------------------------

design <- build_design(run_id, panels = panels)
summ <- design_summary(design)
cat("\n== design ==\n")
print(summ$by_panel)
cat("\ntotal cells: ", summ$total, "\n", sep = "")

## Randomise execution order, seeded by run_id so it is reproducible.
##
## Cells run in design order would put every 1-thread measurement at the start
## and every 32-thread one at the end, so any thermal or contention drift over
## the run would be perfectly confounded with the thing being measured.
set.seed(as.integer(strtoi(substr(digest::digest(run_id, algo = "xxhash32",
                                                 serialize = FALSE), 1, 7),
                           16L) %% .Machine$integer.max))
design <- design[sample.int(length(design))]

## --- striping ---------------------------------------------------------------

stripe <- opt("stripe", NULL)
if (!is.null(stripe)) {
  parts <- as.integer(strsplit(stripe, "/")[[1]])
  if (length(parts) != 2L || any(is.na(parts)) || parts[1] < 0 ||
      parts[1] >= parts[2]) {
    stop("--stripe must be i/n with 0 <= i < n", call. = FALSE)
  }
  ## Interleaved, not contiguous: a contiguous block would hand one task all
  ## the 200k frontier cells and another all the 1k ones.
  idx <- seq_along(design)
  design <- design[(idx - 1L) %% parts[2] == parts[1]]
  cat("stripe ", parts[1], "/", parts[2], " -> ", length(design),
      " cells\n", sep = "")
}

if (dry_run) {
  cat("\n-- dry run, nothing executed --\n")
  print(summ$by_package)
  quit(save = "no", status = 0L)
}

## Applied AFTER shuffling and striping, so a limited run is a random sample of
## the panel rather than its first few cells in design order -- which would be
## all one package on all one dataset.
if (!is.na(limit) && limit > 0L && limit < length(design)) {
  design <- design[seq_len(limit)]
  cat("limited to ", length(design), " cells\n", sep = "")
}

## --- guards -----------------------------------------------------------------

cap <- bench_capability()
## strict = TRUE is the cluster default: a serial sparseDist against
## TBB-threaded competitors is not a comparison, so refuse rather than produce
## numbers nobody can use. --allow-no-openmp downgrades it to a warning for
## the pipeline rehearsal; run-cell.R then blanks every measurement and marks
## the row "dry_run", which validate_row() enforces.
guard_timing(cap, strict = !allow_serial)
if (!can_time(cap)) {
  cat("\n!! No OpenMP: this is a PIPELINE REHEARSAL. Every cell will record\n",
      "!! status \"dry_run\" with no timing or memory. Nothing produced here\n",
      "!! is usable as a measurement.\n\n", sep = "")
}

## Pin BLAS at run time as well as through the environment. arma::cor sits
## under pearson and covariance, and a threaded BLAS there would put
## parallelism into the nominally single-threaded column.
if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
  RhpcBLASctl::blas_set_num_threads(1L)
}

## --- execute ----------------------------------------------------------------

done_already <- function(spec) {
  f <- file.path(out_dir, paste0(spec$cell_id, ".rds"))
  if (!file.exists(f)) return(FALSE)
  row <- tryCatch(validate_row(readRDS(f)), error = function(e) NULL)
  if (is.null(row)) return(FALSE)
  identical(row$run_id, spec$run_id) && row$status %in% c("ok", "dry_run")
}

n <- length(design)
t0 <- Sys.time()
skipped <- 0L
tally <- character(0)

for (i in seq_len(n)) {
  spec <- design[[i]]
  if (resume && done_already(spec)) { skipped <- skipped + 1L; next }

  row <- tryCatch(run_cell(spec, bench_root = root, out_dir = out_dir,
                           cap_sec = cap_sec),
                  error = function(e) {
                    ## A harness-level failure, distinct from a cell failing:
                    ## record it and keep going rather than losing the rest of
                    ## the stripe to one bad spec.
                    message("run_cell failed for ", spec$cell_id, ": ",
                            conditionMessage(e))
                    NULL
                  })
  st <- if (is.null(row)) "harness_error" else row$status
  tally <- c(tally, st)

  el <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
  ## rep is shown because otherwise consecutive replicates of one cell look
  ## like the driver repeating itself.
  cat(sprintf("[%d/%d] %-9s %-14s %-10s %-28s r%-2d t%-3d %6.1f min\n",
              i, n, st, spec$package, spec$method,
              substr(spec$dataset_id, 1, 28), as.integer(spec$rep),
              as.integer(spec$threads), el))
}

cat("\n== summary ==\n")
if (skipped) cat("skipped (already done): ", skipped, "\n", sep = "")
if (length(tally)) print(table(status = tally))
cat("elapsed: ", round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1),
    " min\n", sep = "")

## Pool whatever exists so far. Non-strict: a stripe finishing while siblings
## are mid-write should not abort on a partial directory.
res <- tryCatch(pool_results(out_dir, strict = FALSE),
                error = function(e) NULL)
if (!is.null(res) && nrow(res)) {
  atomic_saveRDS(res, file.path(root, "results",
                                paste0("results-", run_id, ".rds")))
  cat("pooled ", nrow(res), " rows -> results/results-", run_id, ".rds\n",
      sep = "")
}
