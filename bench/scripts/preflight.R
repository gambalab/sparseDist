#!/usr/bin/env Rscript
## ---------------------------------------------------------------------------
## preflight.R -- run before submitting anything. Nothing gets queued until
## this exits 0.
##
##   Rscript bench/scripts/preflight.R && sbatch bench/scripts/run.sbatch
##
## That idiom is why the exit status matters: a script that prints FAIL and
## exits 0 submits the job anyway. A twelve-hour job dying at hour nine on a
## missing Bioconductor dependency is the standard way a benchmark gets lost,
## and everything below is cheap and catches exactly that.
## ---------------------------------------------------------------------------

## Root derived from THIS script's own path, not the working directory:
## normalizePath("..") resolves differently depending on where you invoke from,
## which silently breaks source() and is maddening to diagnose.
script_path <- function() {
  ca  <- commandArgs(trailingOnly = FALSE)
  hit <- grep("^--file=", ca, value = TRUE)
  if (length(hit)) normalizePath(sub("^--file=", "", hit[1]), mustWork = FALSE)
  else NA_character_
}
sp   <- script_path()
root <- Sys.getenv("BENCH_ROOT", unset = if (is.na(sp)) getwd()
                                        else dirname(dirname(sp)))

failures  <- character()
warnings_ <- character()

line <- function(label, status, detail = "")
  cat(sprintf("  %-38s %-6s %s\n", label, status, detail))

## required = FALSE for conditions that legitimately differ between the Mac dry
## run and the cluster: reported, but non-blocking.
check <- function(label, passed, detail = "", required = TRUE) {
  passed <- isTRUE(passed)
  line(label, if (passed) "PASS" else if (required) "FAIL" else "WARN", detail)
  if (!passed) {
    if (required) failures  <<- c(failures, label)
    else          warnings_ <<- c(warnings_, label)
  }
  invisible(passed)
}

cat("\nBENCH_ROOT: ", root, "\n", sep = "")

cat("\n== harness modules ==\n")
loader <- file.path(root, "R", "load-harness.R")
if (!file.exists(loader)) {
  cat("  FATAL: cannot find ", loader, "\n", sep = "")
  quit(save = "no", status = 1L)
}
source(loader)

## Put the benchmark library on the path BEFORE checking for packages --
## otherwise everything installed into it reports as missing, which is a path
## problem wearing the costume of an install problem.
lib <- use_bench_library(root)
check("benchmark library", !is.na(lib),
      if (is.na(lib))
        "not found: set BENCH_LIB, or run setup.R to record it in the manifest"
      else lib)

present <- harness_status(root)
for (f in names(present)) check(f, present[[f]],
                                if (present[[f]]) "" else "MISSING")

## Source all six, not merely check existence: a syntax error or a failed
## top-level dependency inside the adapters is exactly the class of fault that
## otherwise appears once per cell, after submission.
sourced <- tryCatch({load_harness(root, require_all = TRUE); TRUE},
                    error = function(e) {
                      cat("      ", conditionMessage(e), "\n"); FALSE
                    })
check("all modules source cleanly", sourced)
if (!sourced) {
  cat("\nPreflight FAILED: harness will not load.\n")
  quit(save = "no", status = 1L)
}

cat("\n== environment ==\n")
cap <- bench_capability()
check("R version", TRUE, cap$r_version)
check("host / OS", TRUE, paste(cap$host, cap$os))
check("BLAS",      TRUE, cap$blas)

## Expected to fail on macOS, a correctness-only platform by design. On the
## cluster the driver's guard_timing(strict = TRUE) makes it a hard stop.
check("sparseDist OpenMP", cap$openmp,
      if (cap$openmp) paste0("spec ", cap$omp_spec)
      else "absent -- correctness only, no timings",
      required = FALSE)

## Only meaningful when OpenMP is present. Without it ompInfoCpp() reports
## num_procs = 1 by construction -- that is the "kernels run serially" signal,
## not a cpuset restriction -- so comparing it against the hardware count would
## announce a confinement that does not exist. Which it did, on the Mac.
if (isTRUE(cap$openmp)) {
  check("OpenMP procs / hardware", TRUE,
        paste0(cap$num_procs, " / ", cap$hw_threads,
               if (isTRUE(cap$num_procs < cap$hw_threads))
                 "  (CONFINED: cpuset/cgroup)" else ""))
} else {
  check("OpenMP procs / hardware", TRUE,
        paste0("n/a without OpenMP; hardware reports ", cap$hw_threads,
               " threads"))
}

check("peak RSS (VmHWM)", cap$has_vmhwm,
      if (cap$has_vmhwm) "" else "Linux only; memory table unavailable",
      required = FALSE)
check("RSS reset (clear_refs)", cap$can_reset,
      if (cap$can_reset) "" else "peak_rss_delta_mb will be NA",
      required = FALSE)
check("wall-clock cap backend", timeout_backend() != "none", timeout_backend())

sl <- slurm_provenance()
check("SLURM provenance", TRUE,
      if (is.na(sl$slurm_job_id)) "not under SLURM (interactive)"
      else paste0("job ", sl$slurm_job_id))

cat("\n== packages ==\n")
pkgs <- c("sparseDist", "Matrix",
          "proxyC", "text2vec", "coop", "parallelDist", "philentropy",
          "BiocNeighbors", "bluster", "dbscan",
          "processx", "RhpcBLASctl", "digest", "jsonlite",
          "igraph", "irlba")
for (p in pkgs) {
  have <- requireNamespace(p, quietly = TRUE)
  check(p, have,
        if (have) as.character(utils::packageVersion(p)) else "NOT INSTALLED")
}

cat("\n== child process ==\n")
## The parent finding a package proves nothing about the children: they launch
## with --vanilla and their own environment. thread_env() propagates
## .libPaths(), and this is the assertion that it actually works -- the exact
## failure that made the first api-dump report every package as absent.
check("child subprocess can load packages",
      tryCatch(child_can_load(c("sparseDist", "proxyC")),
               error = function(e) FALSE),
      "launches Rscript --vanilla and requires sparseDist + proxyC")

## The PARENT seeing 64 processors proves nothing about the children, and the
## children are what run every cell. An OMP_PLACES setting once made libgomp
## fall back to two places in a batch step only -- max_threads capped at 2, a
## whole scaling panel flat, and nothing upstream noticed. Ask a child directly.
child_omp <- function(want = 8L) {
  ## num_procs, NOT max_threads. omp_get_max_threads() just echoes
  ## OMP_NUM_THREADS, so it reported 8 when we asked for 8 even though the
  ## runtime could see only 2 processors -- this check passed while the panel
  ## it was meant to protect ran at 2 threads.
  code <- 'cat(sparseDist:::ompInfoCpp()$num_procs)'
  env <- thread_env(want)
  out <- tryCatch({
    if (requireNamespace("processx", quietly = TRUE)) {
      processx::run(file.path(R.home("bin"), "Rscript"),
                    args = c("--vanilla", "-e", code),
                    env = c("current", env), error_on_status = FALSE)$stdout
    } else NA_character_
  }, error = function(e) NA_character_)
  suppressWarnings(as.integer(trimws(out)))
}
if (isTRUE(cap$openmp)) {
  got <- child_omp(8L)
  ok_child <- isTRUE(got >= cap$num_procs)
  check("child OpenMP sees all processors", ok_child,
        if (is.na(got)) "could not query child"
        else paste0("child num_procs = ", got, ", parent = ", cap$num_procs))
  if (!ok_child && !is.na(got)) {
    cat("      A child confined below the parent means OMP_PLACES is set\n",
        "      somewhere in the job script or submitting shell; processx\n",
        "      inherits it via env = c(\"current\", ...).\n", sep = "")
  }
}

cat("\n== smoke test ==\n")
smoke <- tryCatch({
  set.seed(1)
  X <- abs(Matrix::rsparsematrix(200, 40, density = 0.3))
  colnames(X) <- paste0("c", seq_len(ncol(X)))
  d  <- sparseDist::sparseDist(X, ncores = 1, verbose = FALSE)
  nn <- sparseDist::sparseKNN(X, k = 5, method = "cosine",
                              ncores = 1, verbose = FALSE)
  g  <- sparseDist::sparseSNN(nn, prune = 1/15, ncores = 1, verbose = FALSE)
  stopifnot(dim(d) == c(40, 40), dim(nn$idx) == c(40, 5), dim(g) == c(40, 40))
  TRUE
}, error = function(e) {cat("      ", conditionMessage(e), "\n"); FALSE})
check("sparseDist end-to-end pipeline", smoke)

cat("\n== harness self-tests ==\n")
## Each of these encodes a bug that actually occurred. They are regressions,
## not decoration.

check("result row round-trips",
      tryCatch({
        validate_row(new_result_row(cell_id = "x", run_id = "y",
                                    status = "skipped")); TRUE
      }, error = function(e) {cat("      ", conditionMessage(e), "\n"); FALSE}))

spec_ok <- list(run_id = "r", cell_id = "c", experiment = "pairwise",
                package = "p", method = "cosine", dataset_id = "d",
                threads = 1, phase = "kernel", rep = 1)

## Assert the SPECIFIC error. Accepting any error would let an unrelated
## failure inside validate_spec() masquerade as the check passing.
check("spec validator rejects bad phase",
      tryCatch({
        validate_spec(modifyList(spec_ok, list(phase = "kernal"))); FALSE
      }, error = function(e) grepl("^Invalid phase", conditionMessage(e))))

## 3e9 is the interesting case: as.integer() returns NA, so a naive comparison
## errors with "missing value where TRUE/FALSE needed" rather than the message.
check("spec validator rejects huge thread count",
      tryCatch({
        validate_spec(modifyList(spec_ok, list(threads = 3e9))); FALSE
      }, error = function(e) grepl("^threads must be", conditionMessage(e))))

## A numeric run_id passes a presence check but breaks the parent's
## identical() comparison against the child's as.character() round-trip -- so
## every good result gets discarded and replaced by a synthesised failure row.
check("spec validator rejects numeric run_id",
      tryCatch({
        validate_spec(modifyList(spec_ok, list(run_id = 123))); FALSE
      }, error = function(e)
        grepl("non-empty character field", conditionMessage(e))))

## on.exit() belongs to a function frame; local() supplies one explicitly
## rather than relying on the evaluator's top-level context.
check("atomic publish replaces and round-trips",
      tryCatch(local({
        f <- tempfile("atomic-check-", fileext = ".rds")
        on.exit(unlink(f), add = TRUE)
        saveRDS(list(a = 0), f)
        atomic_saveRDS(list(a = 1), f)
        identical(readRDS(f)$a, 1)
      }), error = function(e) FALSE))

id_base <- list(experiment = "knn", package = "sparseDist", method = "cosine",
                dataset_id = "d", threads = 1, phase = "kernel", rep = 1, k = 20)

check("cell ids distinguish block_size",
      tryCatch(!identical(
        make_cell_id(modifyList(id_base, list(block_size = 1024))),
        make_cell_id(modifyList(id_base, list(block_size = 4096)))),
        error = function(e) FALSE))

## run-cell.R seeds the RNG before get_dataset(), so two specs differing only
## in seed can benchmark different generated data. Without seed in the identity
## they share a filename and the second overwrites the first.
check("cell ids distinguish seed",
      tryCatch(!identical(
        make_cell_id(modifyList(id_base, list(seed = 1))),
        make_cell_id(modifyList(id_base, list(seed = 2)))),
        error = function(e) FALSE))

## Sanitisation maps both variants to the same safe string; only the hash over
## the canonical spec keeps them apart.
check("cell ids survive variant sanitisation",
      tryCatch(!identical(
        make_cell_id(modifyList(id_base, list(variant = "dist=FALSE/full=TRUE"))),
        make_cell_id(modifyList(id_base, list(variant = "dist=FALSE full=TRUE")))),
        error = function(e) FALSE))

## as.integer(2.5) is 2, so without validation INSIDE canonicalisation a
## malformed k = 2.5 would take the id of the valid k = 2 cell and overwrite
## its result. make_cell_id() runs before validate_spec(), so this is the only
## place that can catch it.
check("cell ids reject fractional k",
      tryCatch({
        make_cell_id(modifyList(id_base, list(k = 2.5))); FALSE
      }, error = function(e) grepl("'k'", conditionMessage(e))))

check("cell ids reject fractional block_size",
      tryCatch({
        make_cell_id(modifyList(id_base, list(block_size = 1024.5))); FALSE
      }, error = function(e) grepl("'block_size'", conditionMessage(e))))

## The converse: semantically identical specs must NOT produce two ids, or one
## logical cell appears twice in a plot.
check("cell ids canonicalise integer counts",
      tryCatch(identical(
        make_cell_id(modifyList(id_base, list(k = 20,  block_size = 1024))),
        make_cell_id(modifyList(id_base, list(k = 20L, block_size = 1024L)))),
        error = function(e) FALSE))

check("cell ids canonicalise absent vs NA",
      tryCatch(identical(
        make_cell_id(modifyList(id_base, list(threads = 1,  variant = NULL))),
        make_cell_id(modifyList(id_base, list(threads = 1L,
                                              variant = NA_character_)))),
        error = function(e) FALSE))

check("new_cell_spec normalises optional fields",
      tryCatch({
        s <- new_cell_spec(run_id = "r", experiment = "pairwise",
                           package = "sparseDist", method = "cosine",
                           dataset_id = "d", threads = 1, phase = "kernel",
                           rep = 1)
        all(!is.null(s$k), is.na(s$k), !is.null(s$block_size),
            !is.null(s$variant), !is.null(s$seed), nzchar(s$cell_id))
      }, error = function(e) {cat("      ", conditionMessage(e), "\n"); FALSE}))

check("new_cell_spec rejects fractional k at construction",
      tryCatch({
        new_cell_spec(run_id = "r", experiment = "knn", package = "sparseDist",
                      method = "cosine", dataset_id = "d", threads = 1,
                      phase = "kernel", rep = 1, k = 2.5)
        FALSE
      }, error = function(e) grepl("'k'", conditionMessage(e))))

check("dry_run rows cannot carry measurements",
      tryCatch({
        validate_row(new_result_row(cell_id = "x", run_id = "y",
                                    status = "dry_run", peak_rss_total_mb = 12))
        FALSE
      }, error = function(e) grepl("dry_run", conditionMessage(e))))

cat("\n== adapters ==\n")
## Every declared adapter must at least CONSTRUCT. A typo in a switch() or a
## missing package function would otherwise surface once per cell at run time.
bad <- character()
for (i in seq_len(nrow(ADAPTER_TABLE))) {
  r <- ADAPTER_TABLE[i, ]
  ok <- tryCatch({
    get_adapter(r$package, r$experiment, r$method,
                list(threads = 1L, k = 20L, block_size = 256L, variant = NA))
    TRUE
  }, error = function(e) FALSE)
  if (!ok) bad <- c(bad, paste0(r$package, "/", r$experiment, "/", r$method))
}
check("all declared adapters construct", length(bad) == 0,
      if (length(bad)) paste(bad, collapse = ", ")
      else paste(nrow(ADAPTER_TABLE), "adapters"))

## --- verdict ---------------------------------------------------------------
if (length(warnings_)) {
  cat("\nNon-blocking warnings:\n  - ",
      paste(warnings_, collapse = "\n  - "), "\n", sep = "")
}
if (length(failures)) {
  cat("\nPreflight FAILED:\n  - ",
      paste(failures, collapse = "\n  - "), "\n", sep = "")
  quit(save = "no", status = 1L)
}
cat("\nPreflight complete: PASS\n")
quit(save = "no", status = 0L)
