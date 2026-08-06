## ---------------------------------------------------------------------------
## 30-align.R -- does every competitor compute the same thing we do?
##
##   run_alignment()   -> tidy data.frame, one row per comparison
##   run_conventions() -> tidy data.frame of degenerate-input behaviour
##
## This runs BEFORE any timing and is the precondition for it. A benchmark
## comparing two functions that differ semantically produces a plausible number
## rather than an error, which no amount of schema validation catches -- so
## every competitor is checked against sparseDist on small inputs first, and
## the differences that are conventions rather than errors are reported
## separately instead of being averaged into an accuracy figure.
##
## Runs IN-PROCESS, not through run_cell(): there is no timing to protect, the
## inputs are small, and a single traceback is far easier to debug than a
## directory of per-cell error rows.
##
## THREE THINGS ARE SEPARATED OUT, because folding any of them into one error
## number destroys the number:
##
##   DEGENERATE COLUMNS. sparseDist defines cos(0,0) = J(0,0) = 1 and
##     cos(0,x) = 0; proxyC and text2vec return 0, coop returns NaN. These are
##     documented choices. Pairs involving a degenerate column are masked out
##     and reported in their own table.
##
##   THE DIAGONAL. proxyC computes euclidean from the expansion
##     ||x||^2 + ||y||^2 - 2<x,y>, which loses precision exactly where the
##     distance is zero: its self-distances come out around 3e-7 rather than 0.
##     Our merge walk computes (x_i - y_i)^2 directly and is exact there. That
##     is a real accuracy result, so it gets its own column instead of being
##     reported as a failure.
##
##   TIES. At low density most column pairs are disjoint, so hundreds of
##     candidates sit at identical distance and any two exact kNN
##     implementations legitimately return different index sets. Recall is
##     therefore computed against OUR OWN distance to each returned neighbour,
##     not against our index list.
## ---------------------------------------------------------------------------

## Pass thresholds. Both are reported; a comparison passes on either, because
## euclidean and covariance produce large values where an absolute tolerance is
## meaningless, while cosine and Jaccard sit in [0, 1] where a relative one is
## unstable near zero.
ALIGN_ATOL <- 1e-10
ALIGN_RTOL <- 1e-8

## --- degenerate columns -----------------------------------------------------

## Which columns make a pair's value convention-dependent rather than defined.
##
## Emptiness is the usual case, but correlation and covariance additionally
## break on any CONSTANT column -- the variance is zero, so the coefficient is
## 0/0 regardless of whether the column holds non-zeros.
degenerate_cols <- function(X, method) {
  empty <- as.vector(Matrix::colSums(X != 0) == 0)
  if (!method %in% c("pearson", "covariance")) return(empty)
  mu  <- as.vector(Matrix::colMeans(X))
  ex2 <- as.vector(Matrix::colMeans(X * X))
  empty | (ex2 - mu^2) <= .Machine$double.eps
}

align_mask <- function(X, method) {
  ok <- !degenerate_cols(X, method)
  outer(ok, ok, "&")
}

## --- comparison primitives --------------------------------------------------

## Off-diagonal and diagonal reported separately: see the header note on
## proxyC's euclidean self-distances.
compare_matrices <- function(ours, theirs, mask) {
  if (!identical(dim(ours), dim(theirs))) {
    return(list(n_compared = 0, max_abs = NA_real_, max_rel = NA_real_,
                diag_max_abs = NA_real_,
                note = paste0("dimension mismatch: ",
                              paste(dim(ours), collapse = "x"), " vs ",
                              paste(dim(theirs), collapse = "x"))))
  }
  offdiag <- mask & !diag(TRUE, nrow(mask))
  ondiag  <- mask &  diag(TRUE, nrow(mask))

  summarise <- function(sel) {
    a <- ours[sel]; b <- theirs[sel]
    bad <- !is.finite(a) | !is.finite(b)
    a <- a[!bad]; b <- b[!bad]
    if (!length(a)) return(list(n = 0, abs = NA_real_, rel = NA_real_,
                                nbad = sum(bad)))
    d <- abs(a - b); sc <- pmax(abs(a), abs(b))
    list(n = length(a), abs = max(d),
         rel = max(ifelse(sc > 0, d / sc, 0)), nbad = sum(bad))
  }
  off <- summarise(offdiag)
  dg  <- summarise(ondiag)

  ## A non-finite value surviving the mask is a degenerate case we failed to
  ## anticipate; say so rather than dropping it silently with na.rm = TRUE.
  note <- if (off$nbad + dg$nbad > 0)
            paste0(off$nbad + dg$nbad, " non-finite cell(s) after masking")
          else NA_character_

  list(n_compared = off$n, max_abs = off$abs, max_rel = off$rel,
       diag_max_abs = dg$abs, note = note)
}

## kNN agreement, tie-aware.
##
## Index equality is the wrong test. Ties are broken by rule -- ours is "lower
## column index wins" -- and at low density hundreds of candidates share a
## distance, so two CORRECT exact implementations disagree wholesale. Measuring
## index overlap there reports a recall around 0.57 for a search that is
## perfectly exact.
##
## Instead: a returned neighbour counts as correct if OUR OWN distance to it is
## no greater than our k-th distance for that query. That is the definition of
## exactness, and it is indifferent to which member of a tied group was chosen.
## Needs the full reference distance matrix, which is affordable because the
## alignment datasets are small by design.
compare_knn <- function(ours, theirs, ref_dist, keep_query,
                        compare_dist = TRUE, tol = 1e-9) {
  oi <- ours$idx; ti <- theirs$idx
  if (!identical(dim(oi), dim(ti))) {
    return(list(n_compared = 0, recall = NA_real_, exact = NA_real_,
                max_abs = NA_real_, max_rel = NA_real_,
                diag_max_abs = NA_real_, note = "kNN index dimension mismatch"))
  }
  k <- ncol(oi)
  qs <- which(keep_query)
  if (!length(qs)) {
    return(list(n_compared = 0, recall = NA_real_, exact = NA_real_,
                max_abs = NA_real_, max_rel = NA_real_,
                diag_max_abs = NA_real_,
                note = "every query column is degenerate"))
  }

  rec <- vapply(qs, function(i) {
    d <- ref_dist[i, ]; d[i] <- Inf              # self is never a neighbour
    thresh <- sort(d, partial = k)[k]
    mean(d[ti[i, ]] <= thresh + tol)
  }, numeric(1))
  exact <- mean(oi[qs, , drop = FALSE] == ti[qs, , drop = FALSE], na.rm = TRUE)

  ab <- NA_real_; rl <- NA_real_
  if (compare_dist && !is.null(ours$dist) && !is.null(theirs$dist)) {
    a <- ours$dist[qs, , drop = FALSE]; b <- theirs$dist[qs, , drop = FALSE]
    ok <- is.finite(a) & is.finite(b)
    if (any(ok)) {
      d <- abs(a[ok] - b[ok]); sc <- pmax(abs(a[ok]), abs(b[ok]))
      ab <- max(d); rl <- max(ifelse(sc > 0, d / sc, 0))
    }
  }
  list(n_compared = length(qs) * k, recall = mean(rec), exact = exact,
       max_abs = ab, max_rel = rl, diag_max_abs = NA_real_,
       note = if (compare_dist) NA_character_
              else "distances not comparable for this metric")
}

## SNN agreement: the edge SET and the weights on it, separately. Two graphs
## can carry identical weights and still disagree structurally -- which is
## exactly what edges = "knn" versus bluster does.
compare_snn <- function(ours, theirs) {
  ours   <- methods::as(ours, "dgCMatrix")
  theirs <- methods::as(theirs, "dgCMatrix")
  if (!identical(dim(ours), dim(theirs))) {
    return(list(n_compared = 0, edge_jaccard = NA_real_, max_abs = NA_real_,
                max_rel = NA_real_, diag_max_abs = NA_real_,
                note = "SNN dimension mismatch"))
  }
  eo <- which(as.vector(ours != 0)); et <- which(as.vector(theirs != 0))
  ej <- length(intersect(eo, et)) / max(1L, length(union(eo, et)))
  keep <- union(eo, et)
  a <- as.vector(ours)[keep]; b <- as.vector(theirs)[keep]
  d <- abs(a - b); sc <- pmax(abs(a), abs(b))
  list(n_compared = length(keep), edge_jaccard = ej,
       max_abs = if (length(d)) max(d) else NA_real_,
       max_rel = if (length(d)) max(ifelse(sc > 0, d / sc, 0)) else NA_real_,
       diag_max_abs = NA_real_, note = NA_character_)
}

## --- the suite --------------------------------------------------------------

align_row <- function(...) {
  base <- list(dataset_id = NA_character_, experiment = NA_character_,
               method = NA_character_, package = NA_character_,
               variant = NA_character_, n_compared = NA_real_,
               max_abs = NA_real_, max_rel = NA_real_,
               diag_max_abs = NA_real_, recall = NA_real_, exact = NA_real_,
               edge_jaccard = NA_real_, pass = NA, status = NA_character_,
               note = NA_character_)
  given <- list(...)
  base[names(given)] <- given
  as.data.frame(base, stringsAsFactors = FALSE)
}

verdict <- function(cmp) {
  if (!isTRUE(cmp$n_compared > 0)) return(NA)
  isTRUE(cmp$max_abs <= ALIGN_ATOL) || isTRUE(cmp$max_rel <= ALIGN_RTOL)
}

run_alignment <- function(ids = ids_alignment(), k = 10L, verbose = TRUE) {
  competitors <- ADAPTER_TABLE[ADAPTER_TABLE$package != "sparseDist", ]
  out <- list()

  for (id in ids) {
    dat <- tryCatch(get_dataset(id), error = function(e) NULL)
    if (is.null(dat)) {
      out[[length(out) + 1L]] <- align_row(
        dataset_id = id, status = "error",
        note = "dataset unavailable (10x source not fetched?)")
      next
    }
    if (verbose) message("== ", id, "  (", dat$n_rows, " x ", dat$n_cols,
                         ", density ", signif(dat$density, 3), ")")

    ## Reference distance matrices for the tie-aware kNN recall, one per
    ## method, computed once per dataset.
    ref_cache <- new.env(parent = emptyenv())
    ref_full <- function(X, method) {
      key <- paste0(method, "-", ncol(X))
      if (!is.null(ref_cache[[key]])) return(ref_cache[[key]])
      m <- as_dense(sparseDist::sparseDist(X, method = method, full = TRUE,
                                           diag = TRUE, dist = TRUE,
                                           ncores = 1L, verbose = FALSE))
      ref_cache[[key]] <- m
      m
    }

    ## kNN comparisons run on a dataset with degenerate columns REMOVED.
    ##
    ## Unlike the pairwise case, a degenerate column cannot be masked out after
    ## the fact: it corrupts the search itself. BiocNeighbors' "Cosine"
    ## normalises each point to unit length, and an all-zero point normalises
    ## to NaN -- which then sorts as the NEAREST neighbour of every query. At
    ## density 0.01 three such columns dragged recall to 0.71 for a search that
    ## is otherwise exact. Dropping them makes the comparison well posed; the
    ## behaviour itself is worth reporting as a limitation rather than hidden.
    knn_dropped <- 0L
    knn_data <- function(method) {
      keep <- !degenerate_cols(dat$X, method)
      knn_dropped <<- sum(!keep)
      if (all(keep)) return(dat)
      describe_dataset(paste0(dat$id, "-nodegen"),
                       dat$X[, keep, drop = FALSE])
    }

    for (i in seq_len(nrow(competitors))) {
      r <- competitors[i, ]
      if (!method_applicable(r$package, r$method, dat)) next
      ## dbscan reports shared-neighbour COUNTS, not Jaccard weights: a
      ## workflow comparator with no numerical alignment to check.
      if (identical(r$package, "dbscan")) next

      spec <- list(threads = 1L, k = k, block_size = 256L, variant = NA)
      res <- tryCatch({
        ref_ad <- get_adapter("sparseDist", r$experiment,
                              if (identical(r$experiment, "snn")) "jaccard"
                              else r$method, spec)
        ad <- get_adapter(r$package, r$experiment, r$method, spec)

        ## kNN uses the degenerate-free view; everything else uses the mask.
        kd <- if (identical(r$experiment, "knn")) knn_data(r$method) else dat
        ours   <- ref_ad$canonical(ref_ad$run(ref_ad$prepare(kd)))
        theirs <- ad$canonical(ad$run(ad$prepare(kd)))

        if (identical(r$experiment, "pairwise")) {
          cmp <- compare_matrices(ours, theirs, align_mask(dat$X, r$method))
          align_row(dataset_id = id, experiment = r$experiment,
                    method = r$method, package = r$package,
                    n_compared = cmp$n_compared, max_abs = cmp$max_abs,
                    max_rel = cmp$max_rel, diag_max_abs = cmp$diag_max_abs,
                    pass = verdict(cmp), status = "ok", note = cmp$note)
        } else if (identical(r$experiment, "knn")) {
          ## BiocNeighbors normalises and applies Euclidean for "Cosine", so
          ## its distance VALUES are on a different scale even where the
          ## neighbours agree exactly (probe 6). Indices are compared; values
          ## are not.
          cd <- !(identical(r$package, "BiocNeighbors") &&
                  identical(r$method, "cosine"))
          cmp <- compare_knn(ours, theirs, ref_full(kd$X, r$method),
                             rep(TRUE, ncol(kd$X)), compare_dist = cd)
          nt <- cmp$note
          if (knn_dropped > 0L) {
            nt <- paste0(if (is.na(nt)) "" else paste0(nt, "; "),
                         knn_dropped, " degenerate column(s) dropped")
          }
          align_row(dataset_id = id, experiment = r$experiment,
                    method = r$method, package = r$package,
                    n_compared = cmp$n_compared, recall = cmp$recall,
                    exact = cmp$exact, max_abs = cmp$max_abs,
                    max_rel = cmp$max_rel,
                    pass = isTRUE(cmp$recall >= 1 - 1e-12),
                    status = "ok", note = nt)
        } else {
          cmp <- compare_snn(ours, theirs)
          align_row(dataset_id = id, experiment = r$experiment,
                    method = r$method, package = r$package,
                    variant = "edges=shared", n_compared = cmp$n_compared,
                    edge_jaccard = cmp$edge_jaccard, max_abs = cmp$max_abs,
                    max_rel = cmp$max_rel,
                    pass = isTRUE(cmp$edge_jaccard >= 1 - 1e-12) &&
                           isTRUE(cmp$max_abs <= ALIGN_ATOL),
                    status = "ok", note = cmp$note)
        }
      }, error = function(e) {
        align_row(dataset_id = id, experiment = r$experiment,
                  method = r$method, package = r$package,
                  status = "error", note = substr(conditionMessage(e), 1, 300))
      })
      out[[length(out) + 1L]] <- res
    }
  }
  res <- do.call(rbind, out)
  rownames(res) <- NULL
  res
}

## --- conventions ------------------------------------------------------------

## What each package returns for degenerate input. NOT an accuracy measure --
## these are documented choices, and the manuscript should carry them as a
## small qualitative table rather than folding them into an error column.
run_conventions <- function() {
  X <- as_dgc(as.matrix(data.frame(a = c(1, 1, 0), b = c(0, 0, 0),
                                   c = c(0, 0, 0))))
  dat <- describe_dataset("conventions", X)
  spec <- list(threads = 1L, k = 2L, block_size = 256L, variant = NA)

  rows <- list()
  for (pkg in c("sparseDist", "proxyC", "coop", "text2vec", "parallelDist")) {
    for (m in c("cosine", "binary")) {
      if (!adapter_exists(pkg, "pairwise", m)) next
      v <- c(NA_real_, NA_real_, NA_real_); note <- NA_character_
      tryCatch({
        ad <- get_adapter(pkg, "pairwise", m, spec)
        M <- suppressWarnings(ad$canonical(ad$run(ad$prepare(dat))))
        v <- c(M[1, 2], M[2, 3], M[2, 2])
      }, error = function(e) {
        ## Erroring on an all-zero column is itself a convention worth
        ## recording -- silently reporting NA would hide it.
        note <<- substr(conditionMessage(e), 1, 120)
      })
      rows[[length(rows) + 1L]] <- data.frame(
        package = pkg, method = m,
        nonempty_vs_empty = v[1], empty_vs_empty = v[2], empty_diagonal = v[3],
        note = note, stringsAsFactors = FALSE)
    }
  }
  res <- do.call(rbind, rows); rownames(res) <- NULL; res
}

## --- reporting --------------------------------------------------------------

align_report <- function(res) {
  ok   <- sum(res$pass %in% TRUE)
  fail <- sum(res$pass %in% FALSE)
  err  <- sum(res$status %in% "error")
  cat(sprintf("\nalignment: %d pass, %d FAIL, %d error, %d rows\n",
              ok, fail, err, nrow(res)))

  ## Reported, never a failure: our exact self-distances against proxyC's
  ## expansion-based ones.
  dg <- res[is.finite(res$diag_max_abs) & res$diag_max_abs > ALIGN_ATOL, ]
  if (nrow(dg)) {
    cat("\nDiagonal (self-distance) differences -- reported, not failures:\n")
    print(dg[, c("dataset_id", "package", "method", "diag_max_abs")])
  }

  bad <- res[res$pass %in% FALSE | res$status %in% "error", ]
  if (nrow(bad)) {
    cat("\nNeeds attention:\n")
    print(bad[, c("dataset_id", "package", "experiment", "method",
                  "max_abs", "max_rel", "recall", "edge_jaccard", "note")])
  }
  invisible(res)
}
