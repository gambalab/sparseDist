## ---------------------------------------------------------------------------
## 01-capability.R -- what can this machine legitimately measure?
##
## The central rule: a build without OpenMP may run correctness checks but must
## never emit a timing. On stock Apple clang sparseDist is serial whatever
## `ncores` says, while proxyC and parallelDist reach TBB perfectly well -- so a
## timing taken there compares our one thread against their many and silently
## understates the package. Enforcing this in code rather than by memory is the
## whole point of the file.
## ---------------------------------------------------------------------------

bench_capability <- function() {
  info <- tryCatch(sparseDist:::ompInfoCpp(),
                   error = function(e) list(openmp = NA, spec = NA_integer_,
                                            max_threads = NA_integer_,
                                            num_procs = NA_integer_,
                                            hw_threads = NA_integer_))

  blas <- tryCatch({
    ext <- utils::sessionInfo()$BLAS
    if (is.null(ext)) "unknown" else basename(ext)
  }, error = function(e) "unknown")

  list(
    openmp      = isTRUE(info$openmp),
    omp_spec    = if (is.null(info$spec)) NA_real_ else as.numeric(info$spec),
    max_threads = as.numeric(info$max_threads),
    num_procs   = as.numeric(info$num_procs),
    hw_threads  = as.numeric(info$hw_threads),
    ## Linux only. Absence is not fatal -- it downgrades memory measurement to
    ## unavailable, which on the Mac is expected.
    has_vmhwm   = file.exists("/proc/self/status"),
    can_reset   = file.exists("/proc/self/clear_refs"),
    os          = paste(Sys.info()[["sysname"]], Sys.info()[["release"]]),
    host        = Sys.info()[["nodename"]],
    r_version   = paste0(R.version$major, ".", R.version$minor),
    blas        = blas
  )
}

## TRUE when this machine may produce publishable timings.
can_time <- function(cap = bench_capability()) isTRUE(cap$openmp)

## Call at the top of any driver that records timings.
##   strict = TRUE  -> stop outright (the cluster)
##   strict = FALSE -> warn; cells then record status "dry_run" with all
##                     timing and RSS fields blank (the Mac)
guard_timing <- function(cap = bench_capability(), strict = TRUE) {
  if (can_time(cap)) {
    ## A cpuset-restricted allocation is not an error, but sizing anything from
    ## hw_threads under one is, so surface the discrepancy loudly and early.
    if (is.finite(cap$num_procs) && is.finite(cap$hw_threads) &&
        cap$num_procs < cap$hw_threads) {
      message("NOTE: OpenMP sees ", cap$num_procs, " processors but the ",
              "hardware has ", cap$hw_threads, ". This process is confined ",
              "(cpuset/cgroup). Thread counts above ", cap$num_procs,
              " will oversubscribe.")
    }
    return(invisible(TRUE))
  }
  msg <- paste0("This build of sparseDist has no OpenMP support, so `ncores` ",
                "is inert and every sparseDist timing would be serial while ",
                "TBB-based competitors are not. Timings from this machine are ",
                "not comparable.")
  if (strict) stop(msg, call. = FALSE)
  warning(msg, call. = FALSE)
  invisible(FALSE)
}

## Environment for a child process: thread pinning AND library visibility.
##
## LIBRARY PATH. Children are launched with --vanilla, which implies
## --no-environ and therefore ignores .Renviron. If the benchmark packages live
## in a project library (./bench-lib) rather than the system one, the child
## sees only the default .libPaths() and every cell fails with "there is no
## package called 'proxyC'" -- an error that looks like an adapter fault and is
## not. Propagating the PARENT's .libPaths() explicitly removes the dependency
## on the operator having exported R_LIBS by hand.
##
## THREADS. Every runtime in the comparison set is pinned through a different
## variable, and BLAS separately: fastCorr calls arma::cor, so a threaded BLAS
## underneath would inject hidden parallelism into the nominally
## single-threaded column and contaminate the scaling curve.
thread_env <- function(threads, lib_paths = .libPaths()) {
  threads <- as.character(as.integer(threads))
  c(R_LIBS                     = paste(lib_paths, collapse = .Platform$path.sep),
    OMP_NUM_THREADS            = threads,
    OMP_PROC_BIND              = "close",
    OMP_PLACES                 = "cores",
    RCPP_PARALLEL_NUM_THREADS  = threads,   # proxyC, parallelDist (TBB)
    ## BLAS pinned to 1 unconditionally: we measure each package's own
    ## parallelism, and nested BLAS threads would both distort the scaling
    ## curve and oversubscribe the node.
    OPENBLAS_NUM_THREADS       = "1",
    MKL_NUM_THREADS            = "1",
    VECLIB_MAXIMUM_THREADS     = "1",
    OMP_DYNAMIC                = "FALSE")
}

## Can a freshly launched child actually see the benchmark packages?
##
## Verifies the R_LIBS propagation above end to end, rather than trusting it.
## This is the exact failure that made the first api-dump run report every
## package as absent, so preflight asserts it.
child_can_load <- function(pkgs = c("sparseDist", "proxyC"),
                           rscript = file.path(R.home("bin"), "Rscript")) {
  code <- sprintf(
    'q(status = if (all(vapply(c(%s), requireNamespace, logical(1), quietly = TRUE))) 0L else 1L)',
    paste0('"', pkgs, '"', collapse = ", "))
  env <- thread_env(1)
  status <- tryCatch({
    if (requireNamespace("processx", quietly = TRUE)) {
      processx::run(rscript, args = c("--vanilla", "-e", code),
                    env = c("current", env), error_on_status = FALSE)$status
    } else {
      system2(rscript, args = c("--vanilla", "-e", shQuote(code)),
              env = paste0(names(env), "=", env),
              stdout = FALSE, stderr = FALSE)
    }
  }, error = function(e) 1L)
  identical(as.integer(status), 0L)
}
