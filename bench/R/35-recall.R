## ---------------------------------------------------------------------------
## 35-recall.R -- accuracy of the approximate kNN backends.
##
##   run_knn_recall(ids, k) -> tidy data.frame
##
## WHY THIS IS A SEPARATE STEP. The kNN timing cells do not retain their
## output, so nothing in the timing pipeline can compute recall. Without this
## the manuscript would carry Annoy and HNSW timings with no accuracy figure --
## exactly the "approximate methods presented as simply faster equivalents"
## that the benchmark brief warns against. Approximate timings are meaningless
## without the recall that bought them.
##
## TWO RECALL DEFINITIONS, both reported, because they answer different
## questions and disagree in the regime we care about.
##
##   recall_set   |their k INTERSECT our k| / k. The figure used throughout the
##                ANN literature, so it is what a reader will compare against.
##                It punishes tie-breaking differences that are not errors.
##
##   recall_tie   fraction of their neighbours whose TRUE distance is within
##                our k-th distance. This is the definition of exactness and is
##                indifferent to which member of a tied group was returned. At
##                low density, where hundreds of candidates share a distance,
##                recall_set understates an exact method badly -- it read 0.57
##                for BiocNeighbors' exact backend during alignment, which is
##                perfectly exact.
##
## recall_tie needs the full n x n reference distance matrix, so it is computed
## only up to RECALL_MAX_N. Above that recall_set is reported alone, with the
## caveat that it is a lower bound.
##
## VARIANTS ARE SWEPT, not taken at their defaults. HnswParam defaults to
## ef.search = 10, narrower than k = 20, which caps recall near 0.94 for
## reasons that have nothing to do with the algorithm. Pairing each variant
## with its timing from the knn panel gives the recall-versus-time curve that
## is the only fair way to report an approximate method.
## ---------------------------------------------------------------------------

## 20k columns is a 3.2 GB dense reference matrix. Beyond that the tie-aware
## calculation costs more than the search being measured.
RECALL_MAX_N <- 20000L

## k values for the sweep.
##
## Recall at fixed k cannot distinguish two very different failures. HNSW sits
## at 0.957 on the 2000-dimensional HVG matrix and barely moves with ef.search
## -- and (k-1)/k is 0.95 at k = 20. If that plateau is ONE systematically
## missed neighbour per query, recall will track (k-1)/k as k varies: 0.80 at
## k = 5, 0.98 at k = 50. If it is genuine approximation error it will sit near
## 0.96 throughout. The two predictions diverge sharply and one sweep decides.
BENCH_K_SWEEP <- c(5L, 10L, 20L, 50L, 100L)

## Variants for the sweep, chosen so ef.search >= k at every k tested.
## Benchmarking HNSW with a search beam narrower than the number of neighbours
## requested measures the configuration, not the algorithm -- bare "hnsw" uses
## ef.search = max(k, 10) for exactly that reason.
BENCH_K_SWEEP_VARIANTS <- c("exact", "annoy-200", "hnsw", "hnsw-200")

recall_row <- function(...) {
  base <- list(dataset_id = NA_character_, n_cols = NA_real_,
               density = NA_real_, method = NA_character_,
               package = NA_character_, variant = NA_character_,
               k = NA_real_, n_queries = NA_real_,
               recall_set = NA_real_, recall_tie = NA_real_,
               recall_if_one_miss = NA_real_,
               status = NA_character_, note = NA_character_)
  given <- list(...); base[names(given)] <- given
  as.data.frame(base, stringsAsFactors = FALSE)
}

## Standard ANN recall: overlap of index sets.
recall_set_overlap <- function(ours_idx, theirs_idx) {
  k <- ncol(ours_idx)
  mean(vapply(seq_len(nrow(ours_idx)), function(i) {
    length(intersect(ours_idx[i, ], theirs_idx[i, ])) / k
  }, numeric(1)))
}

## Default ids cover both the size axis and the DIMENSION axis, because recall
## turned out to depend far more on the latter.
default_recall_ids <- function() {
  ## unique(): "pbmc-rna-hvg-n<BENCH_DIM_N>" is both the top of the size ladder
  ## and the 2000-dimensional endpoint of the dimension series, so without this
  ## the most expensive dataset is computed twice and appears twice in the
  ## output table.
  unique(c(ids_size_ladder("pbmc-rna-hvg", BENCH_SIZES),
           sprintf("pbmc-rna-%s-n%d", BENCH_DIM_FORMS, BENCH_DIM_N)))
}

run_knn_recall <- function(ids = default_recall_ids(),
                           ks = BENCH_K,
                           methods = c("cosine", "euclidean"),
                           variants = BENCH_ANN_VARIANTS,
                           verbose = TRUE) {
  out <- list()
  for (id in ids) {
    dat <- tryCatch(get_dataset(id), error = function(e) NULL)
    if (is.null(dat)) {
      out[[length(out) + 1L]] <- recall_row(dataset_id = id, status = "error",
                                            note = "dataset unavailable")
      next
    }
    ## Degenerate columns are dropped for the same reason as in the alignment
    ## suite: BiocNeighbors normalises each point for Cosine, an all-zero point
    ## normalises to NaN, and it then sorts as everyone's nearest neighbour.
    ## That corrupts the search rather than the comparison, so it cannot be
    ## masked out afterwards.
    if (verbose) message("== ", id, " (", dat$n_cols, " columns)")

    for (mt in methods) {
      keep <- !degenerate_cols(dat$X, mt)
      kd <- if (all(keep)) dat else
        describe_dataset(paste0(dat$id, "-nodegen"),
                         dat$X[, keep, drop = FALSE])
      dropped <- sum(!keep)

      ## The reference distance matrix depends only on (dataset, method), not
      ## on k -- so it is computed ONCE here and reused across the whole k
      ## sweep. Recomputing it per k would dominate the run.
      ref <- NULL
      if (kd$n_cols <= RECALL_MAX_N) {
        ref <- tryCatch(as_dense(sparseDist::sparseDist(
          kd$X, method = mt, full = TRUE, diag = TRUE, dist = TRUE,
          ncores = 1L, verbose = FALSE)), error = function(e) NULL)
      }

      for (kk in ks) {
        spec <- list(threads = 1L, k = kk, block_size = 256L, variant = NA)
        ours <- tryCatch({
          ad <- get_adapter("sparseDist", "knn", mt, spec)
          ad$canonical(ad$run(ad$prepare(kd)))
        }, error = function(e) NULL)
        if (is.null(ours)) {
          out[[length(out) + 1L]] <- recall_row(
            dataset_id = id, method = mt, k = kk, status = "error",
            note = "reference sparseKNN failed")
          next
        }

        for (vr in variants) {
          res <- tryCatch({
            sp <- spec; sp$variant <- vr
            ad <- get_adapter("BiocNeighbors", "knn", mt, sp)
            theirs <- ad$canonical(ad$run(ad$prepare(kd)))

            rset <- recall_set_overlap(ours$idx, theirs$idx)
            rtie <- NA_real_
            if (!is.null(ref)) {
              cmp <- compare_knn(ours, theirs, ref, rep(TRUE, kd$n_cols),
                                 compare_dist = FALSE)
              rtie <- cmp$recall
            }
            nt <- c(if (dropped > 0L)
                      paste0(dropped, " degenerate column(s) dropped"),
                    if (is.null(ref)) "too large for tie-aware recall")
            recall_row(dataset_id = id, n_cols = kd$n_cols,
                       density = kd$density, method = mt,
                       package = "BiocNeighbors", variant = vr, k = kk,
                       n_queries = kd$n_cols, recall_set = rset,
                       recall_tie = rtie, status = "ok",
                       ## The one-missed-neighbour hypothesis in numeric form,
                       ## so the comparison is in the table rather than left to
                       ## the reader to compute.
                       recall_if_one_miss = (kk - 1) / kk,
                       note = if (length(nt)) paste(nt, collapse = "; ")
                              else NA_character_)
          }, error = function(e) {
            recall_row(dataset_id = id, method = mt, package = "BiocNeighbors",
                       variant = vr, k = kk, status = "error",
                       note = substr(conditionMessage(e), 1, 200))
          })
          out[[length(out) + 1L]] <- res
        }
      }
    }
  }
  res <- do.call(rbind, out); rownames(res) <- NULL; res
}

recall_report <- function(res) {
  cat("\n== kNN recall ==\n")
  ok <- res[res$status %in% "ok", , drop = FALSE]
  if (!nrow(ok)) {cat("(nothing computed)\n"); return(invisible(res))}
  print(ok[, c("dataset_id", "method", "variant", "k", "recall_tie",
               "recall_if_one_miss")])

  ## The exact backend is the control: if it does not come back at 1.0 on the
  ## tie-aware measure, the comparison itself is wrong and no approximate
  ## number from this run means anything.
  ctl <- ok[ok$variant %in% "exact" & is.finite(ok$recall_tie), ]
  if (nrow(ctl) && any(ctl$recall_tie < 1 - 1e-9)) {
    cat("\nWARNING: the EXACT backend did not reach tie-aware recall 1.0.\n",
        "That is a fault in the comparison, not in BiocNeighbors, and the\n",
        "approximate figures from this run should not be used.\n", sep = "")
  }
  invisible(res)
}
