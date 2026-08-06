## ---------------------------------------------------------------------------
## 02-runner.R -- run one benchmark cell in a fresh R process.
##
## WHY A SUBPROCESS. Three reasons, each sufficient alone:
##   1. Peak RSS is a per-process high-water mark. Two cells in one process and
##      the second inherits the first's peak.
##   2. Thread-pool environment variables are read once, at runtime init. TBB
##      and libgomp do not re-read OMP_NUM_THREADS mid-session, so a scaling
##      sweep inside one process would silently reuse the first thread count.
##   3. A cell that OOMs or segfaults kills only itself. In a 256 GB frontier
##      experiment, cells dying is the expected case, not the exception.
##
## WHY NOT bench::mark() FOR MEMORY. Its mem_alloc instruments R's allocator
## only. Our dense arma::mat accumulator and TBB's arena are both invisible to
## it, producing a memory table wrong in our own favour -- worse than none.
## ---------------------------------------------------------------------------

## --- resident set size (Linux only) ----------------------------------------

## Current RSS, sampled immediately before the measured phase so the peak can
## be read as growth rather than as an absolute the reader must guess at.
rss_current_mb <- function() {
  if (!file.exists("/proc/self/status")) return(NA_real_)
  lines <- tryCatch(readLines("/proc/self/status", warn = FALSE),
                    error = function(e) character(0))
  hit <- grep("^VmRSS:", lines, value = TRUE)
  if (!length(hit)) return(NA_real_)
  kb <- suppressWarnings(as.numeric(gsub("[^0-9]", "", hit[1])))
  if (is.na(kb)) NA_real_ else kb / 1024
}

## Reset the kernel's peak-RSS counter (Linux >= 4.0).
##
## IMPORTANT: this sets the high-water mark to the process's CURRENT RSS, not
## to zero. The subsequent VmHWM therefore still includes R, the loaded
## namespaces and the dataset. It is a peak TOTAL, and only the difference from
## the sampled baseline approximates the phase's own cost.
rss_reset <- function() {
  if (!file.exists("/proc/self/clear_refs")) return(FALSE)
  tryCatch({
    con <- file("/proc/self/clear_refs", open = "w")
    on.exit(close(con), add = TRUE)
    writeLines("5", con)
    TRUE
  }, error = function(e) FALSE, warning = function(w) FALSE)
}

rss_peak_mb <- function() {
  if (!file.exists("/proc/self/status")) return(NA_real_)
  lines <- tryCatch(readLines("/proc/self/status", warn = FALSE),
                    error = function(e) character(0))
  hit <- grep("^VmHWM:", lines, value = TRUE)
  if (!length(hit)) return(NA_real_)
  kb <- suppressWarnings(as.numeric(gsub("[^0-9]", "", hit[1])))
  if (is.na(kb)) NA_real_ else kb / 1024
}

## --- timing ----------------------------------------------------------------

## Wall clock is the headline; CPU time comes along because user/elapsed is our
## detector for parallelism we did not ask for.
time_it <- function(expr) {
  cpu0 <- proc.time(); t0 <- Sys.time()
  value <- force(expr)
  t1 <- Sys.time(); cpu1 <- proc.time()
  list(value        = value,
       elapsed_sec  = as.numeric(difftime(t1, t0, units = "secs")),
       cpu_user_sec = unname(cpu1[["user.self"]] - cpu0[["user.self"]]),
       cpu_sys_sec  = unname(cpu1[["sys.self"]]  - cpu0[["sys.self"]]))
}

## --- child invocation ------------------------------------------------------

## Wall-clock cap backends, preferred first. "none" is fatal rather than a
## silent downgrade: an uncapped frontier cell can consume a whole allocation.
timeout_backend <- function() {
  if (requireNamespace("processx", quietly = TRUE)) return("processx")
  if (nzchar(Sys.which("timeout")))  return("timeout")
  if (nzchar(Sys.which("gtimeout"))) return("gtimeout")
  "none"
}

run_cell <- function(spec,
                     bench_root,
                     out_dir,
                     cap_sec  = 3600,
                     rscript  = file.path(R.home("bin"), "Rscript"),
                     backend  = timeout_backend()) {

  validate_spec(spec)
  if (identical(backend, "none")) {
    stop("No wall-clock cap available: install the 'processx' package, or GNU ",
         "coreutils `timeout`. Running uncapped risks a single cell consuming ",
         "the entire job allocation.", call. = FALSE)
  }

  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  out_file <- file.path(out_dir, paste0(spec$cell_id, ".rds"))

  ## Remove any prior result BEFORE launching.
  ##
  ## Without this, a cell that succeeded in an earlier run and then times out
  ## in this one leaves yesterday's file in place; the parent finds it, reads
  ## it, and reports a success that did not happen. That produces believable
  ## but wrong tables, which is the worst failure mode available to us.
  if (file.exists(out_file)) {
    rc <- unlink(out_file)
    if (rc != 0L || file.exists(out_file)) {
      stop("Cannot remove stale result file: ", out_file, call. = FALSE)
    }
  }

  spec_file <- tempfile(fileext = ".rds")
  spec$out_file   <- out_file
  spec$bench_root <- bench_root
  spec$cap_sec    <- cap_sec
  saveRDS(spec, spec_file)
  on.exit(unlink(spec_file), add = TRUE)

  child <- file.path(bench_root, "scripts", "run-cell.R")
  args  <- c("--vanilla", child, spec_file)
  env   <- thread_env(spec$threads)

  t0 <- Sys.time()
  if (identical(backend, "processx")) {
    ## "current" is processx's documented spelling for "inherit, then override
    ## these". Splicing Sys.getenv() instead produces duplicate names for any
    ## variable already set, with undefined precedence.
    res <- processx::run(rscript, args = args, env = c("current", env),
                         timeout = cap_sec, error_on_status = FALSE)
    timed_out  <- isTRUE(res$timeout)
    status     <- res$status          # may be NA for a killed process
    stderr_txt <- res$stderr
  } else {
    bin <- if (identical(backend, "timeout")) "timeout" else "gtimeout"
    err <- tempfile()
    status <- suppressWarnings(system2(
      bin, args = c(as.character(cap_sec), shQuote(rscript), shQuote(args)),
      env = paste0(names(env), "=", env), stdout = FALSE, stderr = err))
    timed_out  <- identical(as.integer(status), 124L)  # coreutils convention
    stderr_txt <- paste(readLines(err, warn = FALSE), collapse = "\n")
    unlink(err)
  }
  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

  ## Read defensively: the file may exist but be truncated if the child died
  ## mid-write, and in principle could belong to another cell.
  child_row <- if (file.exists(out_file)) {
    tryCatch(validate_row(readRDS(out_file), file = basename(out_file)),
             error = function(e) NULL)
  } else NULL

  if (!is.null(child_row) &&
      identical(child_row$cell_id, spec$cell_id) &&
      identical(child_row$run_id,  spec$run_id)) {
    return(invisible(child_row))
  }

  ## No usable row: the child died before publishing one. Classify only as far
  ## as the evidence allows. 137 is SIGKILL, which at these sizes is usually
  ## the OOM killer -- but equally the scheduler or a manual kill, and processx
  ## may report NA for a killed process. Hence "killed"; the OOM determination
  ## is a post-hoc sacct annotation joined on slurm_job_id.
  outcome <- if (timed_out) "timeout"
             else if (is.na(status) || identical(as.integer(status), 137L))
               "killed"
             else "error"

  sl <- slurm_provenance()
  row <- new_result_row(
    cell_id     = spec$cell_id,
    run_id      = spec$run_id,
    timestamp   = format(Sys.time(), "%Y-%m-%dT%H:%M:%OS3Z", tz = "UTC"),
    experiment  = spec$experiment,
    package     = spec$package,
    method      = spec$method,
    dataset_id  = spec$dataset_id,
    k           = spec$k,
    block_size  = spec$block_size,
    variant     = spec$variant,
    threads     = spec$threads,
    phase       = spec$phase,
    rep         = spec$rep,
    ## Parent-side wall clock: how long the process lived, not a kernel timing.
    ## Recorded because it is the only duration available for a dead cell, and
    ## it is what the frontier plot needs to show where the cap bites.
    elapsed_sec = elapsed,
    status      = outcome,
    message     = substr(paste0("exit=", status, "; ", stderr_txt), 1, 2000),
    host        = Sys.info()[["nodename"]],
    slurm_job_id        = sl$slurm_job_id,
    slurm_array_job_id  = sl$slurm_array_job_id,
    slurm_array_task_id = sl$slurm_array_task_id
  )
  atomic_saveRDS(validate_row(row), out_file)
  invisible(row)
}
