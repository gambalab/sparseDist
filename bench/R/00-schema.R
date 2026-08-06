## ---------------------------------------------------------------------------
## 00-schema.R -- the one definition of what a benchmark result row is.
##
## Every experiment, on every machine, emits rows of exactly this shape. Fixing
## it here is what makes the Mac dry run and the cluster run concatenable, and
## what lets the plotting code be written before any real numbers exist.
##
## Rows are stored one-per-cell as RDS and pooled at the end. A crashed or
## killed cell still produces a row with status != "ok" -- absence of a row
## means the harness itself failed, which is a different problem and must stay
## visible as such.
## ---------------------------------------------------------------------------

EXPERIMENT_LEVELS <- c("pairwise", "knn", "snn", "frontier", "scaling",
                       "align", "limits")
PHASE_LEVELS      <- c("kernel", "end_to_end")
STATUS_LEVELS     <- c(
  "ok",       # ran, measured, usable
  "dry_run",  # ran correctly, timings deliberately withheld (no OpenMP)
  "skipped",  # not attempted
  "error",    # raised an R condition
  "timeout",  # exceeded the wall-clock cap
  "killed"    # died by signal: OOM, scheduler, or manual. Deliberately NOT
              # "oom" -- that requires cgroup or sacct evidence, and having the
              # level available would invite someone to populate it by guess.
)
RSS_METHODS <- c("vmhwm", "vmhwm_noreset", "unavailable")

## Fields a dry run must never carry. Blanking them in the child is the
## mechanism; enforcing their absence here is the guarantee.
WITHHELD_ON_DRY_RUN <- c("elapsed_sec", "cpu_user_sec", "cpu_sys_sec",
                         "baseline_rss_mb", "peak_rss_total_mb",
                         "peak_rss_delta_mb")

BENCH_SCHEMA <- c(
  ## --- identity -------------------------------------------------------------
  run_id            = "character",
  cell_id           = "character",
  timestamp         = "character",

  ## --- what was measured ----------------------------------------------------
  experiment        = "character",
  ## Which experiment panel this cell belongs to. Several panels share one
  ## operation -- the density sweep, size ladder and coercion study are all
  ## "pairwise" -- so without this they are indistinguishable at analysis time.
  ## Optional, so cells built before it existed still validate.
  panel             = "character",
  package           = "character",
  pkg_version       = "character",
  method            = "character",
  variant           = "character",

  ## --- on what data ---------------------------------------------------------
  dataset_id        = "character",
  n_rows            = "numeric",
  n_cols            = "numeric",
  density           = "numeric",
  data_class        = "character",
  signed            = "logical",

  ## --- parameters -----------------------------------------------------------
  k                 = "numeric",
  block_size        = "numeric",
  threads           = "numeric",
  phase             = "character",
  rep               = "numeric",

  ## --- measurements ---------------------------------------------------------
  elapsed_sec       = "numeric",
  cpu_user_sec      = "numeric",
  cpu_sys_sec       = "numeric",
  ## cpu_user/elapsed should not materially exceed `threads`. When it does,
  ## something we did not ask for is threading underneath -- a parallel BLAS
  ## inside arma::cor, or an unpinned TBB arena. Cheapest detector we have for
  ## a contaminated single-thread column, hence a first-class field.

  ## Memory, as three separate facts rather than one overloaded number.
  ## Writing 5 to /proc/self/clear_refs resets the high-water mark to CURRENT
  ## RSS, not zero, so VmHWM afterwards still contains R, the loaded namespaces
  ## and the dataset. Naming the total honestly and deriving the delta
  ## separately is the only way this column means what it says.
  baseline_rss_mb   = "numeric",  # process RSS just before the measured phase
  peak_rss_total_mb = "numeric",  # peak TOTAL process RSS during it
  peak_rss_delta_mb = "numeric",  # growth attributable to the phase; NA unless
                                  # the mark could actually be reset
  rss_method        = "character",
  object_size_mb    = "numeric",

  ## --- outcome --------------------------------------------------------------
  status            = "character",
  message           = "character",

  ## --- provenance -----------------------------------------------------------
  host              = "character",
  os                = "character",
  r_version         = "character",
  openmp            = "logical",
  omp_spec          = "numeric",
  num_procs         = "numeric",
  hw_threads        = "numeric",
  blas              = "character",
  ## Join key for scheduler accounting. `sacct -j <id> --format=State,MaxRSS`
  ## is how a "killed" row is later upgraded to a confirmed OOM, and without
  ## these there is nothing to join on.
  slurm_job_id        = "character",
  slurm_array_job_id  = "character",
  slurm_array_task_id = "character",
  harness_sha       = "character",
  seed              = "numeric"
)

new_result_row <- function(...) {
  row <- lapply(BENCH_SCHEMA, function(type) {
    switch(type, character = NA_character_, numeric = NA_real_, logical = NA)
  })
  row <- as.data.frame(row, stringsAsFactors = FALSE)

  given   <- list(...)
  unknown <- setdiff(names(given), names(BENCH_SCHEMA))
  if (length(unknown)) {
    stop("unknown result field(s): ", paste(unknown, collapse = ", "),
         call. = FALSE)
  }
  for (nm in names(given)) {
    value <- given[[nm]]
    ## NULL is the common case: a driver passes spec$k for a non-kNN cell.
    ## Coerce to NA rather than erroring -- this constructor is also used by
    ## the error handler, and an error handler that throws loses the cell.
    if (is.null(value) || length(value) == 0L) value <- NA
    if (length(value) != 1L) {
      stop("field '", nm, "' must be length 1, got ", length(value),
           call. = FALSE)
    }
    row[[nm]] <- switch(BENCH_SCHEMA[[nm]],
                        character = as.character(value),
                        numeric   = as.numeric(value),
                        logical   = as.logical(value))
  }
  row
}

## Strict: exact columns, declared types, controlled vocabularies, and the
## internal consistency conditions that would otherwise let a broken cell pass
## as a good measurement.
validate_row <- function(row, file = NULL) {
  where <- if (is.null(file)) "" else paste0(" [", file, "]")
  bad <- function(...) stop("invalid result row", where, ": ", ...,
                            call. = FALSE)

  if (!is.data.frame(row)) bad("not a data.frame")
  if (nrow(row) != 1L)     bad("expected exactly 1 row, got ", nrow(row))

  missing <- setdiff(names(BENCH_SCHEMA), names(row))
  if (length(missing)) bad("missing field(s): ",
                           paste(missing, collapse = ", "))
  extra <- setdiff(names(row), names(BENCH_SCHEMA))
  if (length(extra))   bad("unexpected field(s): ",
                           paste(extra, collapse = ", "))

  row <- row[names(BENCH_SCHEMA)]

  for (nm in names(BENCH_SCHEMA)) {
    want <- BENCH_SCHEMA[[nm]]
    got  <- row[[nm]]
    ok <- switch(want,
                 character = is.character(got),
                 numeric   = is.numeric(got),
                 logical   = is.logical(got))
    if (!ok) bad("field '", nm, "' should be ", want, ", is ", class(got)[1])
  }

  if (is.na(row$cell_id) || !nzchar(row$cell_id)) bad("cell_id is empty")
  if (is.na(row$run_id)  || !nzchar(row$run_id))  bad("run_id is empty")

  if (!row$status %in% STATUS_LEVELS)
    bad("status '", row$status, "' not in: ",
        paste(STATUS_LEVELS, collapse = ", "))
  if (!is.na(row$phase) && !row$phase %in% PHASE_LEVELS)
    bad("phase '", row$phase, "' not in: ",
        paste(PHASE_LEVELS, collapse = ", "))
  if (!is.na(row$experiment) && !row$experiment %in% EXPERIMENT_LEVELS)
    bad("experiment '", row$experiment, "' not recognised")
  if (!is.na(row$rss_method) && !row$rss_method %in% RSS_METHODS)
    bad("rss_method '", row$rss_method, "' not recognised")

  if (!is.na(row$density) && (row$density < 0 || row$density > 1))
    bad("density out of [0,1]: ", row$density)
  if (!is.na(row$threads) && row$threads < 1)
    bad("threads < 1: ", row$threads)

  ## A row claiming success must carry a usable measurement; otherwise it gets
  ## counted in a mean and drags it silently toward zero.
  if (identical(row$status, "ok") && !isTRUE(is.finite(row$elapsed_sec)))
    bad("status 'ok' but elapsed_sec is not finite")

  ## And a dry run must carry NONE of them. Checking elapsed_sec alone would
  ## still let publishable-looking CPU or RSS numbers through, which is exactly
  ## the leak the dry_run status exists to make impossible.
  if (identical(row$status, "dry_run")) {
    vals <- unlist(row[WITHHELD_ON_DRY_RUN], use.names = FALSE)
    if (any(is.finite(vals)))
      bad("status 'dry_run' must not carry timing or RSS measurements")
  }

  row
}

## Pool per-cell RDS files. Fails loudly, naming the offending file: a silently
## dropped cell is indistinguishable from one never scheduled, which is exactly
## the ambiguity the one-row-per-cell contract exists to prevent.
pool_results <- function(dir, strict = TRUE) {
  files <- sort(list.files(dir, pattern = "\\.rds$", full.names = TRUE))
  empty <- new_result_row(cell_id = "x", run_id = "x",
                          status = "skipped")[0, , drop = FALSE]
  if (!length(files)) return(empty)

  rows <- lapply(files, function(f) {
    tryCatch(validate_row(readRDS(f), file = basename(f)),
             error = function(e) {
               if (strict) stop(conditionMessage(e), call. = FALSE)
               warning("dropping ", basename(f), ": ", conditionMessage(e),
                       call. = FALSE)
               NULL
             })
  })
  rows <- Filter(Negate(is.null), rows)
  if (!length(rows)) return(empty)

  out <- do.call(rbind, rows)

  dupes <- unique(out$cell_id[duplicated(out$cell_id)])
  if (length(dupes)) {
    stop("duplicated cell_id(s) -- results directory mixes runs: ",
         paste(dupes, collapse = ", "), call. = FALSE)
  }
  if (length(unique(out$run_id)) > 1L) {
    warning("pooling ", length(unique(out$run_id)),
            " distinct run_id values; check this is intended.", call. = FALSE)
  }
  rownames(out) <- NULL
  out
}
