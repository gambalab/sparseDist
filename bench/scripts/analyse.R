#!/usr/bin/env Rscript
## ---------------------------------------------------------------------------
## analyse.R -- turn pooled result rows into the QC report, tables and figures.
##
##   Rscript bench/scripts/analyse.R                     # newest results file
##   Rscript bench/scripts/analyse.R --run-id=2026...    # a specific run
##   Rscript bench/scripts/analyse.R --cells=DIR         # pool from cells/
##
## READ THE QC SECTION FIRST. Thread contamination and unstable replicates
## invalidate figures without changing their appearance, so they are printed
## before anything is written, and the script exits non-zero if either fires.
## A plot that looks fine is not evidence that the measurement was sound.
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
source(file.path(root, "R", "40-analysis.R"))

args <- commandArgs(trailingOnly = TRUE)
opt <- function(name, default = NULL) {
  hit <- grep(paste0("^--", name, "="), args, value = TRUE)
  if (!length(hit)) return(default)
  sub(paste0("^--", name, "="), "", hit[1])
}

res_dir <- file.path(root, "results")
cells   <- opt("cells", NULL)
run_id  <- opt("run-id", NULL)

## --- load -------------------------------------------------------------------

if (!is.null(cells)) {
  ## Pooling straight from the per-cell directory is the recovery path: if a
  ## job died before writing its pooled file, the cells are still there.
  cat("pooling from ", cells, "\n", sep = "")
  res <- pool_results(cells, strict = FALSE)
} else {
  files <- list.files(res_dir, pattern = "^results-.*\\.rds$",
                      full.names = TRUE)
  if (!is.null(run_id)) {
    files <- files[grepl(run_id, basename(files), fixed = TRUE)]
  }
  if (!length(files)) stop("no results-*.rds under ", res_dir, call. = FALSE)
  files <- files[order(file.info(files)$mtime, decreasing = TRUE)]
  cat("reading ", basename(files[1]), "\n", sep = "")
  res <- readRDS(files[1])
}

if (!nrow(res)) stop("no result rows", call. = FALSE)

## A pooled directory mixing a Mac rehearsal with cluster measurements would
## otherwise contribute rows that are correct but unusable.
n_dry <- sum(res$status %in% "dry_run")
if (n_dry) {
  cat("note: ", n_dry, " dry_run row(s) present (no timings); excluded from ",
      "all timing summaries.\n", sep = "")
}

hosts <- unique(res$host[!is.na(res$host)])
if (length(hosts) > 1L) {
  cat("WARNING: rows from ", length(hosts), " hosts (",
      paste(hosts, collapse = ", "), "). Timings from different machines are ",
      "not comparable and must not share a figure.\n", sep = "")
}

## --- summarise and check ----------------------------------------------------

summ <- bench_summarise(res)
qc <- qc_report(res, summ)

sc  <- scaling_table(summ)
spd <- speedup_table(summ)
fr  <- frontier_table(res)

cat("\n== frontier ==\n")
if (nrow(fr)) print(fr) else cat("(no ladder/frontier rows)\n")

## --- write ------------------------------------------------------------------

tab_dir <- file.path(res_dir, "tables")
fig_dir <- file.path(res_dir, "figures")
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

wr <- function(x, name) {
  if (is.null(x) || !nrow(x)) return(invisible())
  utils::write.csv(x, file.path(tab_dir, paste0(name, ".csv")),
                   row.names = FALSE)
  cat("  tables/", name, ".csv (", nrow(x), " rows)\n", sep = "")
}
cat("\n== written ==\n")
wr(summ, "summary")
wr(spd,  "speedup")
wr(sc,   "scaling")
wr(fr,   "frontier")
wr(qc$contamination, "qc-thread-contamination")
wr(qc$replication,   "qc-unstable-replicates")
wr(qc$failures,      "qc-failed-cells")

if (requireNamespace("ggplot2", quietly = TRUE)) {
  figs <- build_figures(summ, res, sc)
  for (nm in names(figs)) {
    if (is.null(figs[[nm]])) next
    f <- file.path(fig_dir, paste0(nm, ".pdf"))
    ok <- tryCatch({
      ggplot2::ggsave(f, figs[[nm]], width = 9, height = 6, device = "pdf")
      TRUE
    }, error = function(e) {cat("  figure '", nm, "' failed: ",
                                conditionMessage(e), "\n", sep = ""); FALSE})
    if (ok) cat("  figures/", nm, ".pdf\n", sep = "")
  }
} else {
  cat("  (ggplot2 unavailable; tables only)\n")
}

## --- verdict ----------------------------------------------------------------
##
## Non-zero exit on a QC failure so this can gate the write-up the same way
## preflight gates the run. Failed CELLS are not a QC failure -- in the
## frontier panel they are the result.
bad <- nrow(qc$contamination) > 0 || nrow(qc$replication) > 0
if (bad) {
  cat("\nQC FAILED: see qc-*.csv. Figures were written, but the affected\n",
      "measurements are not trustworthy and should be re-run.\n", sep = "")
  quit(save = "no", status = 1L)
}
cat("\nAnalysis complete: QC clean.\n")
quit(save = "no", status = 0L)
