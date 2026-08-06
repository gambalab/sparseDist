#!/usr/bin/env Rscript
## ---------------------------------------------------------------------------
## recall.R -- accuracy of the approximate kNN backends.
##
##   Rscript bench/scripts/recall.R
##   Rscript bench/scripts/recall.R --sizes=1000,5000
##
## Run this alongside the kNN timing panel. Approximate timings without recall
## are not a result, and the two must come from the same datasets and the same
## k or the pairing is meaningless.
##
## Needs no OpenMP: recall is about what is returned, not how fast, so this is
## meaningful on any machine. It IS quadratic in memory for the tie-aware
## measure, which is why the default sizes stop at 20k.
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
source(file.path(root, "R", "20-design.R"))   # BENCH_K, ids_size_ladder
source(file.path(root, "R", "30-align.R"))    # compare_knn, degenerate_cols
source(file.path(root, "R", "35-recall.R"))

args <- commandArgs(trailingOnly = TRUE)
opt <- function(name, default = NULL) {
  hit <- grep(paste0("^--", name, "="), args, value = TRUE)
  if (!length(hit)) return(default)
  sub(paste0("^--", name, "="), "", hit[1])
}

## --k accepts a comma-separated sweep, e.g. --k=5,10,20,50,100
ks <- as.integer(strsplit(opt("k", as.character(BENCH_K)), ",")[[1]])
## --ids wins; --sizes is the size-axis shortcut; otherwise both axes.
ids <- if (!is.null(opt("ids", NULL))) {
  strsplit(opt("ids"), ",")[[1]]
} else if (!is.null(opt("sizes", NULL))) {
  ids_size_ladder("pbmc-rna-hvg",
                  as.integer(strsplit(opt("sizes"), ",")[[1]]))
} else default_recall_ids()

cat("ids  : ", paste(ids, collapse = ", "), "\n", sep = "")
cat("k    : ", paste(ks, collapse = ", "), "\n", sep = "")

res <- run_knn_recall(ids = ids, ks = ks)
recall_report(res)

res_dir <- file.path(root, "results")
dir.create(file.path(res_dir, "tables"), recursive = TRUE,
           showWarnings = FALSE)
atomic_saveRDS(res, file.path(res_dir, "knn-recall.rds"))
utils::write.csv(res, file.path(res_dir, "tables", "knn-recall.csv"),
                 row.names = FALSE)
cat("\nWritten to results/knn-recall.rds\n")

if (any(res$status %in% "error")) quit(save = "no", status = 1L)
quit(save = "no", status = 0L)
