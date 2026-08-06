#!/usr/bin/env Rscript
## ---------------------------------------------------------------------------
## run-cell.R -- executes exactly one benchmark cell, then exits.
##
## Invoked by run_cell() with one argument: the path to an RDS holding the cell
## spec. Publishes one result row to spec$out_file. Anything going wrong inside
## the workload is caught and recorded as status="error"; only process death
## (timeout, OOM, signal) leaves no row, and the parent synthesises one.
##
## ORDER OF OPERATIONS matters for the memory numbers:
##   load packages -> build data -> sample baseline RSS -> reset high-water
##   mark -> run kernel -> read peak
## The reset sets the mark to CURRENT RSS, not zero, so the peak is a TOTAL and
## only peak-minus-baseline approximates the phase's own cost. Both are
## recorded; neither is called "kernel memory".
## ---------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("usage: run-cell.R <spec.rds>")
spec <- readRDS(args[[1]])

## One authoritative source list, shared with preflight. require_all = TRUE:
## a missing module must fail here rather than surfacing as a mysterious
## "could not find function get_dataset" in every single cell.
source(file.path(spec$bench_root, "R", "load-harness.R"))
load_harness(spec$bench_root, require_all = TRUE)

set.seed(if (is.null(spec$seed)) 1L else as.integer(spec$seed))
cap <- bench_capability()
sl  <- slurm_provenance()

base_row <- function(...) {
  new_result_row(
    run_id     = spec$run_id,      cell_id    = spec$cell_id,
    timestamp  = format(Sys.time(), "%Y-%m-%dT%H:%M:%OS3Z", tz = "UTC"),
    experiment = spec$experiment,  panel      = spec$panel,
    package    = spec$package,
    method     = spec$method,      variant    = spec$variant,
    dataset_id = spec$dataset_id,  k          = spec$k,
    block_size = spec$block_size,
    threads    = spec$threads,     phase      = spec$phase,
    rep        = spec$rep,         seed       = spec$seed,
    host       = cap$host,         os         = cap$os,
    r_version  = cap$r_version,    openmp     = cap$openmp,
    omp_spec   = cap$omp_spec,     num_procs  = cap$num_procs,
    hw_threads = cap$hw_threads,   blas       = cap$blas,
    slurm_job_id        = sl$slurm_job_id,
    slurm_array_job_id  = sl$slurm_array_job_id,
    slurm_array_task_id = sl$slurm_array_task_id,
    harness_sha = spec$harness_sha,
    ...
  )
}

## Alignment cells establish CORRECTNESS and are perfectly valid without
## OpenMP -- the capability policy allows exactly that. Everything else is a
## measurement, and a measurement from a build where our kernels are serial
## while TBB competitors are not is simply not comparable.
is_timing_cell  <- !spec$experiment %in% c("align")
withhold_timing <- is_timing_cell && !can_time(cap)

result <- tryCatch({

  ## ---- 1. data ------------------------------------------------------------
  ## Built inside the child so its construction is never charged to the kernel,
  ## and so a dataset too large for the node fails as this cell rather than
  ## taking the whole driver down.
  dat <- get_dataset(spec$dataset_id)

  ## ---- 2. adapter ---------------------------------------------------------
  ## An adapter is a pair of closures: prepare() performs whatever coercion the
  ## competitor demands (transpose, densify, binarise), run() does the measured
  ## work. That split IS the kernel/end_to_end distinction. Coercion is a real
  ## user cost, and hiding it would flatter whichever package happens to match
  ## our native layout.
  ad <- get_adapter(package = spec$package, experiment = spec$experiment,
                    method = spec$method, spec = spec)

  prepared <- if (identical(spec$phase, "kernel")) ad$prepare(dat) else NULL

  invisible(gc(full = TRUE))
  baseline <- rss_current_mb()
  reset_ok <- rss_reset()

  timed <- time_it({
    if (identical(spec$phase, "kernel")) ad$run(prepared)
    else                                 ad$run(ad$prepare(dat))
  })

  peak_total <- rss_peak_mb()
  peak_delta <- if (reset_ok && is.finite(baseline) && is.finite(peak_total)) {
    max(0, peak_total - baseline)
  } else NA_real_

  ## ---- 3. optional correctness payload ------------------------------------
  ## Alignment runs persist their output for comparison against the reference.
  ## compress = FALSE deliberately: gzipping a multi-GB proximity matrix takes
  ## far longer than computing it, and these files are transient.
  if (isTRUE(spec$save_output) && !is.null(spec$output_file)) {
    atomic_saveRDS(timed$value, spec$output_file, compress = FALSE)
  }

  ## ---- 4. row -------------------------------------------------------------
  ## On a build without OpenMP the workload still RUNS -- that is the point of
  ## the Mac dry run, which exists to prove every adapter and code path works
  ## -- but timing and memory fields are left NA. Blanking rather than merely
  ## flagging means an imperfect filter downstream cannot leak a serial number
  ## into a plot: there is no number to leak. validate_row() enforces it.
  base_row(
    pkg_version    = as.character(utils::packageVersion(spec$package)),
    n_rows         = nrow(dat$X), n_cols = ncol(dat$X),
    density        = dat$density, data_class = class(dat$X)[1],
    signed         = dat$signed,
    elapsed_sec       = if (withhold_timing) NA_real_ else timed$elapsed_sec,
    cpu_user_sec      = if (withhold_timing) NA_real_ else timed$cpu_user_sec,
    cpu_sys_sec       = if (withhold_timing) NA_real_ else timed$cpu_sys_sec,
    baseline_rss_mb   = if (withhold_timing) NA_real_ else baseline,
    peak_rss_total_mb = if (withhold_timing) NA_real_ else peak_total,
    peak_rss_delta_mb = if (withhold_timing) NA_real_ else peak_delta,
    rss_method     = if (is.na(peak_total)) "unavailable"
                     else if (reset_ok) "vmhwm" else "vmhwm_noreset",
    object_size_mb = as.numeric(utils::object.size(timed$value)) / 1024^2,
    status         = if (withhold_timing) "dry_run" else "ok",
    message        = if (withhold_timing)
                       "no OpenMP: ran correctly, timings withheld"
                     else NA_character_
  )

}, error = function(e) {
  base_row(status = "error", message = substr(conditionMessage(e), 1, 2000))
})

atomic_saveRDS(validate_row(result), spec$out_file)
