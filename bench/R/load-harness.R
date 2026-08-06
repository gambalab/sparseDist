## ---------------------------------------------------------------------------
## load-harness.R -- the single authoritative list of harness modules, and the
## one place the benchmark library gets put on the search path.
##
## This file exists to stop those two things from being duplicated across
## preflight.R, run-cell.R and any driver. When such copies drift, preflight
## passes while every benchmark child dies on a missing get_dataset() -- a
## failure that only appears after submission, once per cell.
##
## Sourced directly (it cannot be part of the list it defines); everything else
## goes through load_harness().
## ---------------------------------------------------------------------------

HARNESS_FILES <- c(
  "00-utils.R",      # atomic write, spec validation, cell ids
  "00-schema.R",     # result row definition and validation
  "01-capability.R", # OpenMP probe, timing guard, thread pinning
  "02-runner.R",     # RSS sampling, timing, subprocess cell runner
  "03-datasets.R",   # dataset registry: get_dataset()
  "10-adapters.R"    # package adapters: get_adapter()
)

## Where the benchmark packages live.
##
## They are deliberately installed OUTSIDE the package directory (a library
## inside it makes R CMD build try to ship several hundred megabytes of Boost
## headers), which means no process finds them by default. Relying on the
## operator to export R_LIBS has now failed four separate times -- the api
## dump, the benchmark children, an interactive session, and preflight -- each
## time presenting as "package not installed" rather than as a path problem.
##
## setup.R already records the library it installed into, in the per-host
## manifest. Reading it back is both automatic and correct across machines: the
## Mac and the cluster get their own entry.
##
## Resolution order: BENCH_LIB (explicit override) -> manifest -> nothing.
bench_library <- function(root) {
  lib <- Sys.getenv("BENCH_LIB", unset = "")
  if (nzchar(lib) && dir.exists(lib)) return(normalizePath(lib, mustWork = FALSE))

  mf <- file.path(root, "results",
                  paste0("bench-manifest-", Sys.info()[["nodename"]], ".json"))
  if (file.exists(mf) && requireNamespace("jsonlite", quietly = TRUE)) {
    lib <- tryCatch(jsonlite::fromJSON(mf)$library, error = function(e) NULL)
    if (!is.null(lib) && length(lib) == 1L && !is.na(lib) && dir.exists(lib)) {
      return(normalizePath(lib, mustWork = FALSE))
    }
  }
  NA_character_
}

## Prepend the benchmark library, if one can be found. Returns the path used,
## or NA -- callers report it rather than leaving the search path implicit.
use_bench_library <- function(root) {
  lib <- bench_library(root)
  if (!is.na(lib) && !lib %in% .libPaths()) .libPaths(c(lib, .libPaths()))
  invisible(lib)
}

## require_all = FALSE is for bootstrap situations only (e.g. inspecting the
## schema before the adapters exist). Benchmark execution always demands the
## full set: a missing module must fail here, loudly, not inside a queued job.
load_harness <- function(root, require_all = TRUE) {
  use_bench_library(root)
  loaded <- character(0)
  for (f in HARNESS_FILES) {
    path <- file.path(root, "R", f)
    if (!file.exists(path)) {
      if (require_all) stop("Missing harness file: ", path, call. = FALSE)
      next
    }
    source(path, local = FALSE)
    loaded <- c(loaded, f)
  }
  invisible(loaded)
}

## Which of the expected modules are actually present. Used by preflight to
## report the gap explicitly rather than inferring it from a later failure.
harness_status <- function(root) {
  vapply(HARNESS_FILES,
         function(f) file.exists(file.path(root, "R", f)),
         logical(1))
}
