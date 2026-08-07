## ---------------------------------------------------------------------------
## 40-analysis.R -- from result rows to tables and figures.
##
## THREE RULES THAT SHAPE EVERYTHING HERE
##
##  1. MEDIAN AND MAD, never mean and sd. Benchmark timings are right-skewed:
##     one cell catching a page fault or a scheduler hiccup drags a mean, and
##     with five reps the mean has no defence. The median of five is stable and
##     the MAD says whether the five agreed.
##
##  2. FAILED CELLS ARE DATA in the frontier panel. A "killed" row at 200k
##     columns is the result -- it is where the dense p x p allocation stops
##     fitting. Filtering to status == "ok" before plotting the frontier would
##     delete the finding and leave a curve that simply stops for no stated
##     reason. Every other panel does filter; the frontier must not.
##
##  3. DRY-RUN ROWS CARRY NO TIMINGS by construction, and validate_row()
##     enforces it. They are excluded explicitly anyway, so a Mac rehearsal
##     pooled into a cluster directory cannot quietly contribute NA rows to a
##     median.
## ---------------------------------------------------------------------------

## Everything that identifies a measurement except the replicate index.
GROUP_KEYS <- c("panel", "experiment", "package", "method", "variant",
                "dataset_id", "n_rows", "n_cols", "density", "data_class",
                "k", "block_size", "threads", "phase")

## --- summarisation ----------------------------------------------------------

## Columns bench_summarise() adds on top of GROUP_KEYS.
SUMMARY_COLS <- c("n_reps", "elapsed_med", "elapsed_mad", "elapsed_min",
                  "elapsed_cv", "cpu_user_med", "rss_total_med",
                  "rss_delta_med", "objsize_med")

## A zero-row frame with the SUMMARY shape.
##
## Returning the input schema instead -- which is what an early `ok[0, ]`
## does -- gives a frame with no elapsed_cv column, and every downstream check
## then fails on "undefined columns selected". An empty result must still have
## the right columns.
empty_summary <- function() {
  base <- new_result_row(cell_id = "x", run_id = "y",
                         status = "skipped")[0, GROUP_KEYS, drop = FALSE]
  for (nm in SUMMARY_COLS) base[[nm]] <- numeric(0)
  base
}

bench_summarise <- function(res) {
  ok <- res[res$status %in% "ok", , drop = FALSE]
  if (!nrow(ok)) {
    warning("no rows with status 'ok'; nothing to summarise", call. = FALSE)
    return(empty_summary())
  }
  key <- do.call(paste, c(lapply(GROUP_KEYS, function(k) ok[[k]]), sep = "\r"))

  parts <- split(seq_len(nrow(ok)), key)
  rows <- lapply(parts, function(idx) {
    d <- ok[idx, , drop = FALSE]
    med <- stats::median(d$elapsed_sec)
    out <- d[1, GROUP_KEYS, drop = FALSE]
    out$n_reps        <- length(idx)
    out$elapsed_med   <- med
    ## constant = 1.4826 makes MAD comparable to an sd under normality; kept so
    ## the spread column is interpretable next to the median.
    out$elapsed_mad   <- stats::mad(d$elapsed_sec, constant = 1.4826)
    out$elapsed_min   <- min(d$elapsed_sec)
    out$elapsed_cv    <- if (med > 0) out$elapsed_mad / med else NA_real_
    out$cpu_user_med  <- stats::median(d$cpu_user_sec)
    out$rss_total_med <- stats::median(d$peak_rss_total_mb, na.rm = TRUE)
    out$rss_delta_med <- stats::median(d$peak_rss_delta_mb, na.rm = TRUE)
    out$objsize_med   <- stats::median(d$object_size_mb, na.rm = TRUE)
    out
  })
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

## --- quality control --------------------------------------------------------

## Parallelism we did not ask for.
##
## cpu_user / elapsed approximates the number of cores actually busy. If it
## materially exceeds the requested thread count, something else is threading
## underneath -- a parallel BLAS inside arma::cor, or a TBB arena that
## thread_env() failed to pin. A contaminated single-thread column silently
## flattens the whole scaling curve, so this runs before any figure.
check_thread_contamination <- function(res, tolerance = 1.25) {
  ok <- res[res$status %in% "ok" & is.finite(res$elapsed_sec) &
            res$elapsed_sec > 0.05, , drop = FALSE]
  if (!nrow(ok)) return(ok[0, c("panel", "package", "method", "dataset_id",
                                "threads", "elapsed_sec", "blas"),
                           drop = FALSE])
  ok$cores_busy <- ok$cpu_user_sec / ok$elapsed_sec
  bad <- ok[ok$cores_busy > ok$threads * tolerance, , drop = FALSE]
  bad[order(-bad$cores_busy),
      c("panel", "package", "method", "dataset_id", "threads",
        "cores_busy", "elapsed_sec", "blas")]
}

## Replicates that did not agree. A high spread means the median is not
## trustworthy for that cell and it should be re-run, not quietly plotted.
## min_reps matters when analysing a run still in progress. Execution order is
## randomised, so a partially complete panel has cells with one or two of their
## five replicates done -- and a MAD computed from two observations is not a
## spread estimate, it is a coin flip. Judging those would fail QC on every
## partial run and train everyone to ignore the check.
check_replication <- function(summ, cv_limit = 0.20, min_reps = 3L) {
  cols <- c("panel", "package", "method", "dataset_id", "threads", "n_reps",
            "elapsed_med", "elapsed_mad", "elapsed_cv")
  if (!nrow(summ) || !all(cols %in% names(summ))) return(empty_summary()[, 0])
  bad <- summ[is.finite(summ$elapsed_cv) & summ$elapsed_cv > cv_limit &
              summ$elapsed_med > 0.5 & summ$n_reps >= min_reps, , drop = FALSE]
  bad[order(-bad$elapsed_cv), cols, drop = FALSE]
}

## Cells that never produced a usable measurement, by cause. Expected in the
## frontier panel; anywhere else it is a bug.
check_failures <- function(res) {
  bad <- res[!res$status %in% c("ok", "dry_run"), , drop = FALSE]
  if (!nrow(bad)) return(bad)
  bad[order(bad$panel, bad$package),
      c("panel", "package", "method", "dataset_id", "threads", "status",
        "elapsed_sec", "message")]
}

qc_report <- function(res, summ) {
  cat("\n== QC ==\n")
  cat("rows: ", nrow(res), "   status: ",
      paste(names(table(res$status)), table(res$status), sep = "=",
            collapse = ", "), "\n", sep = "")

  ct <- check_thread_contamination(res)
  cat("\nthread contamination (cpu_user/elapsed > threads): ", nrow(ct), "\n",
      sep = "")
  if (nrow(ct)) print(utils::head(ct, 15))

  rp <- check_replication(summ)
  incomplete <- sum(summ$n_reps < 3L)
  cat("\nunstable replicates (MAD/median > 0.20, >=3 reps): ", nrow(rp),
      "\n", sep = "")
  if (incomplete) {
    cat("  (", incomplete, " cell(s) have fewer than 3 replicates so far and ",
        "were not judged -- expected mid-run)\n", sep = "")
  }
  if (nrow(rp)) print(utils::head(rp, 15))

  fl <- check_failures(res)
  cat("\nfailed cells: ", nrow(fl), "\n", sep = "")
  if (nrow(fl)) print(table(panel = fl$panel, status = fl$status))

  invisible(list(contamination = ct, replication = rp, failures = fl))
}

## --- derived tables ---------------------------------------------------------

## Speed relative to sparseDist, matched on everything else. Ratios rather than
## raw seconds, because the reader's question is "how does this compare", and a
## ratio survives being run on a different machine.
speedup_table <- function(summ) {
  if (!nrow(summ)) return(summ)
  key <- function(d) do.call(paste, c(lapply(
    c("panel", "experiment", "method", "dataset_id", "threads", "phase"),
    function(k) d[[k]]), sep = "\r"))
  ref <- summ[summ$package %in% "sparseDist" & is.na(summ$variant), ,
              drop = FALSE]
  if (!nrow(ref)) return(summ[0, ])
  m <- match(key(summ), key(ref))
  out <- summ
  out$ref_elapsed <- ref$elapsed_med[m]
  ## > 1 means sparseDist is faster.
  out$speedup <- out$ref_elapsed / out$elapsed_med
  out$speedup[out$package %in% "sparseDist" & is.na(out$variant)] <- 1
  out[!is.na(out$speedup), ]
}

## Parallel efficiency: t(1) / (N * t(N)). 1.0 is perfect scaling.
scaling_table <- function(summ) {
  if (!nrow(summ)) return(summ)
  s <- summ[summ$panel %in% "scaling", , drop = FALSE]
  if (!nrow(s)) return(s)
  key <- function(d) paste(d$package, d$method, d$dataset_id, sep = "\r")
  base <- s[s$threads == 1, , drop = FALSE]
  m <- match(key(s), key(base))
  s$t1 <- base$elapsed_med[m]
  s$speedup_vs_1 <- s$t1 / s$elapsed_med
  s$efficiency <- s$speedup_vs_1 / s$threads
  ## Hyperthreads are not cores: efficiency against 64 is not comparable with
  ## efficiency against 32, so it is flagged rather than silently plotted.
  s$hyperthreaded <- s$threads > 32
  s
}

## The frontier: for each package and method, the largest input that COMPLETED,
## and what happened above it.
##
## Built from res, not summ, precisely because summ drops everything that is
## not status "ok" -- and here the failures are the point.
frontier_table <- function(res) {
  f <- res[res$panel %in% c("ladder", "frontier"), , drop = FALSE]
  if (!nrow(f)) return(data.frame())
  key <- paste(f$package, f$method, sep = "\r")
  rows <- lapply(split(seq_len(nrow(f)), key), function(idx) {
    d <- f[idx, , drop = FALSE]
    okd <- d[d$status %in% "ok", , drop = FALSE]
    fail <- d[!d$status %in% c("ok", "dry_run"), , drop = FALSE]
    data.frame(
      package = d$package[1], method = d$method[1],
      max_n_completed = if (nrow(okd)) max(okd$n_cols) else NA_real_,
      min_n_failed = if (nrow(fail)) min(fail$n_cols) else NA_real_,
      first_failure = if (nrow(fail))
        fail$status[which.min(fail$n_cols)] else NA_character_,
      stringsAsFactors = FALSE)
  })
  out <- do.call(rbind, rows); rownames(out) <- NULL
  out[order(-out$max_n_completed, out$package), ]
}

## --- figures ----------------------------------------------------------------

.theme <- function() {
  ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(panel.grid.minor = ggplot2::element_blank(),
                   legend.position = "bottom",
                   strip.text = ggplot2::element_text(face = "bold"))
}

## Where does sparsity-aware traversal stop paying? Log-log, because both axes
## span orders of magnitude and the crossover is the point of the figure.
fig_density <- function(summ) {
  d <- summ[summ$panel %in% "density", , drop = FALSE]
  if (!nrow(d)) return(NULL)
  ggplot2::ggplot(d, ggplot2::aes(density, elapsed_med, colour = package)) +
    ggplot2::geom_line() +
    ggplot2::geom_point(size = 1.2) +
    ggplot2::geom_errorbar(
      ggplot2::aes(ymin = pmax(elapsed_med - elapsed_mad, 1e-4),
                   ymax = elapsed_med + elapsed_mad), width = 0) +
    ggplot2::scale_x_log10() + ggplot2::scale_y_log10() +
    ggplot2::facet_wrap(~ method, scales = "free_y") +
    ggplot2::labs(x = "density (stored non-zeros / total)",
                  y = "elapsed (s), median of reps",
                  title = "Pairwise cost against sparsity",
                  subtitle = "2000 features x 3000 observations, 32 threads") +
    .theme()
}

fig_scaling <- function(sc) {
  if (!nrow(sc)) return(NULL)
  ggplot2::ggplot(sc[!sc$hyperthreaded, ],
                  ggplot2::aes(threads, speedup_vs_1, colour = package)) +
    ## Ideal linear scaling, for reference. Without it the eye reads any
    ## upward curve as good scaling.
    ggplot2::geom_abline(slope = 1, intercept = 0, linetype = 2,
                         colour = "grey60") +
    ggplot2::geom_line() + ggplot2::geom_point(size = 1.2) +
    ggplot2::scale_x_continuous(trans = "log2",
                                breaks = c(1, 2, 4, 8, 16, 32)) +
    ggplot2::scale_y_continuous(trans = "log2") +
    ggplot2::facet_wrap(~ method) +
    ggplot2::labs(x = "threads (physical cores)", y = "speedup vs 1 thread",
                  title = "Parallel scaling",
                  subtitle = paste("dashed = ideal; hyperthreaded 64-thread",
                                   "points excluded, reported separately")) +
    .theme()
}

## Size ladder WITH the failures marked. A curve that simply stops invites the
## question this figure exists to answer.
fig_ladder <- function(summ, res) {
  d <- summ[summ$panel %in% c("ladder", "frontier"), , drop = FALSE]
  if (!nrow(d)) return(NULL)
  fail <- res[res$panel %in% c("ladder", "frontier") &
              !res$status %in% c("ok", "dry_run"), , drop = FALSE]
  p <- ggplot2::ggplot(d, ggplot2::aes(n_cols, elapsed_med, colour = package)) +
    ggplot2::geom_line() + ggplot2::geom_point(size = 1.2) +
    ggplot2::scale_x_log10() + ggplot2::scale_y_log10() +
    ggplot2::facet_wrap(~ method, scales = "free_y") +
    ggplot2::labs(x = "observations (columns)", y = "elapsed (s), median",
                  title = "Cost against problem size",
                  subtitle = "crosses mark cells that did not complete") +
    .theme()
  if (nrow(fail)) {
    fail$y <- max(d$elapsed_med, na.rm = TRUE)
    p <- p + ggplot2::geom_point(data = fail,
                                 ggplot2::aes(n_cols, y, colour = package),
                                 shape = 4, size = 3, inherit.aes = FALSE)
  }
  p
}

## Coercion: what matching each package's preferred layout costs.
fig_coercion <- function(summ) {
  d <- summ[summ$panel %in% "coercion", , drop = FALSE]
  if (!nrow(d)) return(NULL)
  ggplot2::ggplot(d, ggplot2::aes(stats::reorder(package, elapsed_med),
                                  elapsed_med, fill = phase)) +
    ggplot2::geom_col(position = "dodge") +
    ggplot2::facet_wrap(~ method, scales = "free_y") +
    ggplot2::coord_flip() +
    ggplot2::labs(x = NULL, y = "elapsed (s), median",
                  title = "Kernel versus end-to-end",
                  subtitle = paste("end-to-end includes transpose,",
                                   "densification and format conversion")) +
    .theme()
}

## Memory. peak_rss_delta is growth during the measured phase; peak_rss_total
## is the whole process. Both are plotted because neither alone is honest --
## the delta excludes the input the method needs resident, the total includes
## R itself.
fig_memory <- function(summ) {
  d <- summ[summ$panel %in% c("ladder", "frontier") &
            is.finite(summ$rss_total_med), , drop = FALSE]
  if (!nrow(d)) return(NULL)
  ggplot2::ggplot(d, ggplot2::aes(n_cols, rss_total_med, colour = package)) +
    ggplot2::geom_line() + ggplot2::geom_point(size = 1.2) +
    ggplot2::scale_x_log10() + ggplot2::scale_y_log10() +
    ggplot2::facet_wrap(~ method) +
    ggplot2::labs(x = "observations (columns)", y = "peak process RSS (MB)",
                  title = "Peak resident memory",
                  subtitle = "total process RSS during the measured phase") +
    .theme()
}

## Output size is machine-independent, so this figure is valid from the Mac
## rehearsal alone -- no OpenMP required.
fig_objsize <- function(summ) {
  d <- summ[is.finite(summ$objsize_med) & summ$panel %in% "density", ,
            drop = FALSE]
  if (!nrow(d)) return(NULL)
  ggplot2::ggplot(d, ggplot2::aes(density, objsize_med, colour = package)) +
    ggplot2::geom_line() + ggplot2::geom_point(size = 1.2) +
    ggplot2::scale_x_log10() + ggplot2::scale_y_log10() +
    ggplot2::facet_wrap(~ method, scales = "free_y") +
    ggplot2::labs(x = "density", y = "returned object size (MB)",
                  title = "Size of the returned object",
                  subtitle = paste("independent of hardware; a dense p x p",
                                   "result does not shrink with sparsity")) +
    .theme()
}

build_figures <- function(summ, res, sc) {
  list(density = fig_density(summ), scaling = fig_scaling(sc),
       ladder = fig_ladder(summ, res), coercion = fig_coercion(summ),
       memory = fig_memory(summ), objsize = fig_objsize(summ))
}
