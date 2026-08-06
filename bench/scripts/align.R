#!/usr/bin/env Rscript
## ---------------------------------------------------------------------------
## align.R -- run the alignment suite and persist the tables.
##
##   Rscript bench/scripts/align.R                 # synthetic datasets only
##   Rscript bench/scripts/align.R --real          # include the 10x datasets
##
## Alignment needs no OpenMP: it establishes what each package COMPUTES, not
## how fast. So this is the part of the benchmark that is fully meaningful on
## the Mac, and it should be green there before anything is queued on the
## cluster.
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
source(file.path(root, "R", "30-align.R"))

args <- commandArgs(trailingOnly = TRUE)
use_real <- "--real" %in% args

ids <- if (use_real) ids_alignment() else grep("^syn-", ids_alignment(),
                                               value = TRUE)
if (!use_real) {
  message("Synthetic datasets only. Re-run with --real once verify_sources() ",
          "has fetched the 10x matrices.")
}

res_dir <- file.path(root, "results")
dir.create(res_dir, recursive = TRUE, showWarnings = FALSE)

cat("\n=== conventions (degenerate input) ===\n")
conv <- run_conventions()
print(conv)
cat("\nThese are documented CHOICES, not errors. sparseDist defines\n",
    "cos(0,0) = J(0,0) = 1 and cos(0,x) = 0; the alignment mask excludes\n",
    "these pairs so one all-zero column cannot dominate the error column.\n",
    sep = "")
atomic_saveRDS(conv, file.path(res_dir, "alignment-conventions.rds"))

cat("\n=== alignment ===\n")
res <- run_alignment(ids = ids)
atomic_saveRDS(res, file.path(res_dir, "alignment.rds"))
utils::write.csv(res, file.path(res_dir, "alignment.csv"), row.names = FALSE)

align_report(res)
cat("\nWritten to ", file.path(res_dir, "alignment.rds"), "\n", sep = "")

## Exit non-zero on any failure or error, so this can gate a submission the
## same way preflight does.
if (any(res$pass %in% FALSE) || any(res$status %in% "error")) {
  quit(save = "no", status = 1L)
}
quit(save = "no", status = 0L)
