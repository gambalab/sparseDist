## ---------------------------------------------------------------------------
## load-harness.R -- the single authoritative list of harness modules.
##
## This file exists to stop the source list from being duplicated between
## preflight.R and run-cell.R. When those two lists drift, preflight passes
## while every benchmark child dies on a missing get_dataset() -- a failure
## that only appears after submission, once per cell.
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

## require_all = FALSE is for bootstrap situations only (e.g. inspecting the
## schema before the adapters exist). Benchmark execution always demands the
## full set: a missing module must fail here, loudly, not inside a queued job.
load_harness <- function(root, require_all = TRUE) {
  loaded <- character(0)
  for (f in HARNESS_FILES) {
    path <- file.path(root, "R", f)
    if (!file.exists(path)) {
      if (require_all) {
        stop("Missing harness file: ", path, call. = FALSE)
      }
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
