## ---------------------------------------------------------------------------
## 10-adapters.R -- how each package is actually called.
##
##   get_adapter(package, experiment, method, spec)
##     -> list(prepare, run, canonical)
##
## THREE-PART CONTRACT
##   prepare(dat)   whatever coercion the package demands -- transpose,
##                  densify, binarise. Timed only in the "end_to_end" phase.
##   run(prepared)  the measured work, in the package's NATIVE output form.
##   canonical(out) that output converted to the common comparison form. Used
##                  ONLY by the alignment experiment, never inside a timing,
##                  so conversion cost never contaminates a measurement.
##
## Coercion sits in prepare() rather than being folded into run() because it is
## a real user cost: t() on a dgCMatrix is a full CSC rebuild, not a view, and
## hiding it would flatter whichever package happens to match our own layout.
##
## CANONICAL FORMS
##   pairwise  a full symmetric dense matrix. SIMILARITY for binary, cosine,
##             pearson and covariance; DISTANCE for euclidean, manhattan and js.
##             Choosing per method, rather than forcing everything to a
##             distance, avoids inventing a transform for covariance -- which is
##             unbounded and has no natural distance form.
##   knn       list(idx, dist), 1-based, neighbours ascending by distance.
##   snn       symmetric dgCMatrix of edge weights.
##
## Everything here that looks arbitrary was measured. See bench/results/
## api-dump.txt and api-probe2.txt; the reasons are recorded inline.
## ---------------------------------------------------------------------------

## Methods whose canonical form is a similarity rather than a distance.
SIMILARITY_METHODS <- c("binary", "cosine", "pearson", "covariance")
is_similarity <- function(method) method %in% SIMILARITY_METHODS

## proxyC rounds by DEFAULT (digits = 14). Left alone, an accuracy table would
## report proxyC's rounding as sparseDist's numerical error.
##
## digits = Inf is NOT the way to disable it: probe 1 showed it silently
## returns an all-zero matrix -- no error, no warning -- which in an accuracy
## table reads as catastrophic disagreement. 22 is past double precision and
## behaves correctly.
PROXYC_DIGITS <- 22

## --- what exists ------------------------------------------------------------
##
## timing_ok = FALSE marks a cell that is valid for CORRECTNESS but must not
## appear in a timing panel, because something other than the algorithm would
## dominate the number.
ADAPTER_TABLE <- local({
  row <- function(package, experiment, method, timing_ok = TRUE, note = "") {
    data.frame(package = package, experiment = experiment, method = method,
               timing_ok = timing_ok, note = note, stringsAsFactors = FALSE)
  }
  do.call(rbind, list(
    ## ---- sparseDist: the reference for every panel ------------------------
    row("sparseDist", "pairwise", "binary"),
    row("sparseDist", "pairwise", "cosine"),
    row("sparseDist", "pairwise", "euclidean"),
    row("sparseDist", "pairwise", "manhattan"),
    row("sparseDist", "pairwise", "pearson"),
    row("sparseDist", "pairwise", "covariance"),
    row("sparseDist", "pairwise", "js"),
    row("sparseDist", "knn",      "binary"),
    row("sparseDist", "knn",      "cosine"),
    row("sparseDist", "knn",      "euclidean"),
    row("sparseDist", "knn",      "manhattan"),
    row("sparseDist", "snn",      "jaccard"),

    ## ---- proxyC: the principal sparse pairwise competitor -----------------
    row("proxyC", "pairwise", "binary"),
    row("proxyC", "pairwise", "cosine"),
    row("proxyC", "pairwise", "pearson"),
    row("proxyC", "pairwise", "euclidean"),
    row("proxyC", "pairwise", "manhattan"),
    row("proxyC", "pairwise", "js"),

    ## ---- text2vec: row-oriented, and partly broken on Matrix 1.7.5 --------
    row("text2vec", "pairwise", "cosine"),
    row("text2vec", "pairwise", "euclidean"),
    row("text2vec", "pairwise", "binary", timing_ok = FALSE,
        note = paste("sim2(x) fails coercing dsTMatrix under Matrix 1.7.5;",
                     "the sim2(x, x) workaround computes the full",
                     "non-symmetric product, roughly 2x the arithmetic.",
                     "Correctness only -- timing it would report a packaging",
                     "incompatibility as an algorithmic difference.")),

    ## ---- coop: cosine / pcor / covar specialist ---------------------------
    row("coop", "pairwise", "cosine"),
    row("coop", "pairwise", "pearson"),
    row("coop", "pairwise", "covariance"),

    ## ---- parallelDist: multithreaded DENSE baseline -----------------------
    row("parallelDist", "pairwise", "euclidean"),
    row("parallelDist", "pairwise", "manhattan"),
    row("parallelDist", "pairwise", "binary"),

    ## ---- philentropy: JS reference ----------------------------------------
    row("philentropy", "pairwise", "js",
        note = paste("Primarily a correctness reference: it validated that",
                     "ours is sqrt(JSD) in nats. Timed across the simplex",
                     "density sweep as well, since our js kernel densifies",
                     "and is density-insensitive while proxyC's traverses",
                     "CSC -- the crossover is worth measuring.")),

    ## ---- BiocNeighbors: exact and approximate kNN -------------------------
    row("BiocNeighbors", "knn", "cosine"),
    row("BiocNeighbors", "knn", "euclidean"),
    row("BiocNeighbors", "knn", "manhattan"),

    ## ---- bluster: exact SNN comparator ------------------------------------
    row("bluster", "snn", "jaccard"),

    ## ---- dbscan: workflow comparator only ---------------------------------
    row("dbscan", "snn", "jaccard", timing_ok = FALSE,
        note = paste("sNN() weights are shared-neighbour COUNTS, not Jaccard",
                     "coefficients, and it takes data rather than a",
                     "precomputed index. Related workflow, different",
                     "quantity."))
  ))
})

adapter_exists <- function(package, experiment, method) {
  any(ADAPTER_TABLE$package == package &
      ADAPTER_TABLE$experiment == experiment &
      ADAPTER_TABLE$method == method)
}

adapter_timing_ok <- function(package, experiment, method) {
  hit <- ADAPTER_TABLE[ADAPTER_TABLE$package == package &
                       ADAPTER_TABLE$experiment == experiment &
                       ADAPTER_TABLE$method == method, ]
  if (!nrow(hit)) return(FALSE)
  isTRUE(hit$timing_ok[1])
}

## --- applicability ----------------------------------------------------------

## Handing a package input it does not support produces a number that looks
## like a disagreement rather than a limitation. Each exclusion is an OBSERVED
## failure, not a convenience. Used by both the design builder and the
## alignment suite, so the two never disagree about what is runnable.
method_applicable <- function(package, method, dat) {
  ## Jensen-Shannon needs columns that are probability distributions.
  if (identical(method, "js") && !isTRUE(dat$simplex)) return(FALSE)
  ## text2vec l2-normalises before binarising for jaccard, which does not
  ## survive negative entries: on signed input it returns values above 1 and
  ## thousands of non-finite cells. Jaccard is a function of the support only,
  ## so this is text2vec preprocessing, not a definitional difference.
  if (identical(package, "text2vec") && identical(method, "binary") &&
      isTRUE(dat$signed)) return(FALSE)
  TRUE
}

## --- shared helpers ---------------------------------------------------------

as_dense <- function(X) if (is.matrix(X)) X else as.matrix(X)

## Observations into ROWS, for the row-oriented packages.
transpose_obs_to_rows <- function(X) Matrix::t(X)

## Same neighbour indices for every SNN competitor.
##
## The SNN panel measures GRAPH CONSTRUCTION, not neighbour search, so the
## index must be identical across packages and computed outside the timed
## region. bluster::neighborsToSNNGraph() is built to take a precomputed index,
## which is what makes this comparison exact rather than approximate.
snn_indices <- function(dat, spec) {
  nn <- sparseDist::sparseKNN(dat$X, k = spec$k %||% 20L,
                              method = "cosine", include_self = FALSE,
                              ncores = 1L, verbose = FALSE)
  nn$idx
}

guard_input <- function(ok, msg) if (!ok) stop(msg, call. = FALSE)

## --- sparseDist -------------------------------------------------------------

adapter_sparseDist <- function(experiment, method, spec) {
  threads <- as.integer(spec$threads)
  k       <- as.integer(spec$k %||% 20L)

  if (identical(experiment, "pairwise")) {
    ## variant "full=FALSE" measures our native lower-triangle mode. It is NOT
    ## comparable with a competitor returning a full matrix, so the driver only
    ## pairs it against our own full=TRUE run. Note it saves no memory -- the
    ## triangle is still a dense p x p allocation with the upper half left at
    ## zero -- only compute.
    full <- !identical(spec$variant, "full=FALSE")
    dist_out <- !is_similarity(method)
    list(
      prepare = function(dat) {
        ## The JS kernel takes a dense arma::mat; every other method takes CSC.
        if (identical(method, "js")) {
          guard_input(isTRUE(dat$simplex),
                      "js requires columns summing to 1; use a *-simplex-* dataset")
          return(as_dense(dat$X))
        }
        dat$X
      },
      run = function(X) sparseDist::sparseDist(
        X, method = method, full = full, diag = full,
        dist = dist_out, ncores = threads, verbose = FALSE),
      canonical = function(out) {
        m <- as_dense(out)
        dimnames(m) <- NULL
        m
      }
    )
  } else if (identical(experiment, "knn")) {
    list(
      prepare = function(dat) dat$X,
      run = function(X) sparseDist::sparseKNN(
        X, k = k, method = method, dist = TRUE, include_self = FALSE,
        block_size = as.integer(spec$block_size %||% 256L),
        ncores = threads, verbose = FALSE),
      canonical = function(out) list(idx = out$idx, dist = out$dist)
    )
  } else if (identical(experiment, "snn")) {
    ## edges = "shared" is the bluster-compatible edge set, verified to agree
    ## exactly. "knn" is our sparser default and is compared against itself.
    edges <- if (identical(spec$variant, "edges=knn")) "knn" else "shared"
    list(
      prepare = function(dat) snn_indices(dat, spec),
      run = function(idx) sparseDist::sparseSNN(
        idx, include_self = TRUE, prune = 0, edges = edges,
        ncores = threads, verbose = FALSE),
      canonical = function(out) methods::as(out, "dgCMatrix")
    )
  } else stop("sparseDist has no '", experiment, "' adapter", call. = FALSE)
}

## --- proxyC -----------------------------------------------------------------

adapter_proxyC <- function(experiment, method, spec) {
  ## proxyC splits similarities and distances across two functions, so exactly
  ## one of these lookups matches and the other must come back empty.
  ##
  ## NOT map[[method]]: `[[` on a named vector with an absent name ERRORS with
  ## subscript out of bounds -- it does not return NULL -- so `%||%` never sees
  ## it and every proxyC adapter dies on whichever lookup misses.
  lookup <- function(map, key) if (key %in% names(map)) unname(map[[key]]) else NULL

  simil_method <- lookup(c(binary = "jaccard", cosine = "cosine",
                           pearson = "correlation"), method)
  dist_method  <- lookup(c(euclidean = "euclidean", manhattan = "manhattan",
                           js = "jensen"), method)
  if (is.null(simil_method) && is.null(dist_method)) {
    stop("proxyC has no '", method, "' adapter", call. = FALSE)
  }

  list(
    prepare = function(dat) {
      ## Already column-oriented; margin = 2 handles the rest. jaccard is
      ## defined on the sparsity pattern (probe 1: identical results for
      ## weighted and binary input), matching ours -- no binarisation needed.
      ##
      ## The dense datasets (simplex, pca50) still need coercing to CSC, and
      ## that cost belongs here where the end-to-end phase will see it.
      if (!methods::is(dat$X, "sparseMatrix")) return(as_dgc(dat$X))
      dat$X
    },
    run = function(X) {
      if (!is.null(simil_method)) {
        proxyC::simil(X, margin = 2, method = simil_method,
                      ## use_nan = FALSE gives 0 for degenerate columns and
                      ## suppresses the warning. Our convention differs
                      ## (cos(0,0) = 1), which is why the alignment step masks
                      ## those pairs rather than pretending they agree.
                      use_nan = FALSE,
                      ## NOT diag = TRUE: in proxyC that means "compute ONLY
                      ## the diagonal", the opposite of our diag argument.
                      diag = FALSE,
                      digits = PROXYC_DIGITS)
      } else {
        proxyC::dist(X, margin = 2, method = dist_method,
                     diag = FALSE, digits = PROXYC_DIGITS)
      }
    },
    canonical = function(out) {
      m <- as_dense(out); dimnames(m) <- NULL
      ## proxyC's "jensen" is the divergence in NATS; ours is its square root
      ## (probe 3: theirs / ours^2 == 1 exactly, and it matches philentropy at
      ## unit = "log" to the digit).
      if (identical(method, "js")) m <- sqrt(m)
      m
    }
  )
}

## --- text2vec ---------------------------------------------------------------

adapter_text2vec <- function(experiment, method, spec) {
  list(
    prepare = function(dat) {
      ## Row-oriented, so the transpose is mandatory -- and it is a genuine CSC
      ## rebuild, which is exactly why it belongs in prepare() and shows up in
      ## the end-to-end timing.
      Xt <- transpose_obs_to_rows(dat$X)
      ## dist2() refuses sparse input for euclidean ("could be calculated only
      ## for dense matrices of class 'matrix'"), so that method is effectively
      ## dense-only in text2vec. Densifying here puts the cost where it is
      ## measured rather than hiding it.
      if (identical(method, "euclidean")) as_dense(Xt) else Xt
    },
    run = function(Xt) {
      if (identical(method, "cosine")) {
        text2vec::sim2(Xt, method = "cosine", norm = "l2")
      } else if (identical(method, "euclidean")) {
        text2vec::dist2(Xt, method = "euclidean", norm = "none")
      } else if (identical(method, "binary")) {
        ## Explicit y: the symmetric path dies coercing dsTMatrix under Matrix
        ## 1.7.5 (probe 4). This works, but computes the full non-symmetric
        ## product -- hence timing_ok = FALSE in ADAPTER_TABLE. text2vec
        ## binarises internally and warns about it, so the function matches
        ## ours; only the code path is unusable.
        suppressWarnings(text2vec::sim2(Xt, Xt, method = "jaccard"))
      } else stop("text2vec has no '", method, "' adapter", call. = FALSE)
    },
    canonical = function(out) {
      m <- as_dense(out); dimnames(m) <- NULL; m
    }
  )
}

## --- coop -------------------------------------------------------------------

adapter_coop <- function(experiment, method, spec) {
  ## Pearson is pcor(), not pearson() -- there is no coop::pearson.
  fn <- switch(method,
               cosine     = coop::cosine,
               pearson    = coop::pcor,
               covariance = coop::covar,
               stop("coop has no '", method, "' adapter", call. = FALSE))
  ## coop dispatches a real sparse method for cosine (cosine.dgCMatrix) but not
  ## for pcor or covar, which have only default/matrix/data.frame methods. So
  ## the sparse advantage is cosine-only, and the other two must be densified
  ## -- a cost that belongs in the measurement, not hidden inside run().
  needs_dense <- !identical(method, "cosine")
  list(
    prepare = function(dat) if (needs_dense) as_dense(dat$X) else dat$X,
    run = function(X) fn(X),
    canonical = function(out) {
      m <- as_dense(out); dimnames(m) <- NULL; m
    }
  )
}

## --- parallelDist -----------------------------------------------------------

adapter_parallelDist <- function(experiment, method, spec) {
  threads <- as.integer(spec$threads)
  pd_method <- switch(method,
                      euclidean = "euclidean",
                      manhattan = "manhattan",
                      ## stats::dist(method = "binary") is Jaccard DISTANCE on
                      ## the non-zero pattern, the same quantity as ours.
                      binary    = "binary",
                      stop("parallelDist has no '", method, "' adapter",
                           call. = FALSE))
  list(
    prepare = function(dat) {
      ## Dense, observations in rows. On a large sparse input this is the
      ## dominant cost and often the reason the cell dies -- which is the
      ## point of including a dense baseline at all.
      t(as_dense(dat$X))
    },
    run = function(M) parallelDist::parDist(M, method = pd_method,
                                            threads = threads),
    canonical = function(out) {
      m <- as.matrix(out); dimnames(m) <- NULL
      ## parDist only ever returns a DISTANCE, but "binary" is in
      ## SIMILARITY_METHODS -- so the canonical form for it is the Jaccard
      ## coefficient, and comparing parDist's output directly would be
      ## comparing J against 1 - J (a flat max-abs of exactly 1).
      if (is_similarity(method)) 1 - m else m
    }
  )
}

## --- philentropy ------------------------------------------------------------

adapter_philentropy <- function(experiment, method, spec) {
  list(
    prepare = function(dat) {
      guard_input(isTRUE(dat$simplex),
                  "philentropy JSD requires columns summing to 1")
      ## Rows are the vectors being compared.
      t(as_dense(dat$X))
    },
    run = function(P) philentropy::JSD(P, unit = "log", est.prob = NULL),
    canonical = function(out) {
      ## JSD returns the DIVERGENCE; ours is its square root. unit = "log"
      ## puts it in nats -- the default log2 differs by a factor of 1/ln2 and
      ## would read as a systematic error.
      m <- as_dense(out); dimnames(m) <- NULL
      sqrt(m)
    }
  )
}

## --- BiocNeighbors ----------------------------------------------------------

adapter_BiocNeighbors <- function(experiment, method, spec) {
  threads <- as.integer(spec$threads)
  k       <- as.integer(spec$k %||% 20L)
  distance <- switch(method,
                     cosine    = "Cosine",
                     euclidean = "Euclidean",
                     manhattan = "Manhattan",
                     stop("BiocNeighbors has no '", method, "' adapter",
                          call. = FALSE))
  ## Exact by default; the approximate backends are a separate experiment and
  ## must never be presented as simply faster equivalents.
  ##
  ## Normalise first: spec$variant is NA (not NULL) for a default cell, and
  ## switch() on a non-character EXPR does not reliably fall through to the
  ## unnamed default.
  variant <- spec$variant
  variant <- if (is.null(variant) || length(variant) != 1L || is.na(variant))
               "exact" else as.character(variant)

  ## Variant grammar: "<backend>" or "<backend>-<tune>", e.g. "hnsw-50". The
  ## tuning number is ef.search for HNSW and search.mult for Annoy -- the
  ## accuracy knob each backend exposes, swept to give a recall-versus-time
  ## curve rather than one arbitrary point on it.
  bits <- regmatches(variant, regexec(
    "^(exact|kmknn|annoy|hnsw)(?:-([0-9]+))?$", variant))[[1]]
  if (!length(bits)) {
    stop("unrecognised BiocNeighbors variant '", variant, "'", call. = FALSE)
  }
  backend <- bits[2]
  tune <- if (nzchar(bits[3])) as.integer(bits[3]) else NA_integer_

  ## HnswParam DEFAULTS TO ef.search = 10. With k = 20 the search beam is
  ## narrower than the number of neighbours requested, so recall is capped by
  ## configuration rather than by the algorithm -- measured at 0.94 where the
  ## same index with ef.search >= k does far better. Benchmarking an
  ## approximate method at a setting nobody would deploy is as misleading as
  ## presenting it as a free speedup, so the floor is k.
  ef_default <- max(k, 10L)

  param <- switch(backend,
    exact = BiocNeighbors::ExhaustiveParam(distance = distance),
    kmknn = BiocNeighbors::KmknnParam(distance = distance),
    annoy = BiocNeighbors::AnnoyParam(
      distance = distance,
      search.mult = if (is.na(tune)) 50L else tune),
    hnsw  = BiocNeighbors::HnswParam(
      distance = distance,
      ef.search = if (is.na(tune)) ef_default else tune),
    stop("unhandled backend '", backend, "'", call. = FALSE))
  list(
    prepare = function(dat) {
      ## findKNN has no `transposed` argument (only queryKNN does), so points
      ## must be in rows. It accepts a dgCMatrix and returns results identical
      ## to dense input (probe 6).
      transpose_obs_to_rows(dat$X)
    },
    run = function(M) BiocNeighbors::findKNN(M, k = k, BNPARAM = param,
                                             num.threads = threads),
    canonical = function(out) {
      ## Indices are directly comparable -- probe 6 showed exact agreement with
      ## sparseKNN on cosine, order included. DISTANCES are not: BiocNeighbors
      ## normalises and applies Euclidean for "Cosine", so the values differ
      ## even where the neighbours agree. The alignment step compares indices
      ## (and recall@k) for cosine, values only for Euclidean and Manhattan.
      list(idx = out$index, dist = out$distance)
    }
  )
}

## --- bluster ----------------------------------------------------------------

adapter_bluster <- function(experiment, method, spec) {
  threads <- as.integer(spec$threads)
  list(
    prepare = function(dat) snn_indices(dat, spec),
    run = function(idx) bluster::neighborsToSNNGraph(idx, type = "jaccard",
                                                     num.threads = threads),
    canonical = function(out) {
      ## Verified: identical to sparseSNN(edges = "shared") -- max abs
      ## difference 0 over the full adjacency matrix.
      m <- igraph::as_adjacency_matrix(out, attr = "weight", sparse = TRUE)
      methods::as(methods::as(m, "generalMatrix"), "CsparseMatrix")
    }
  )
}

## --- dbscan -----------------------------------------------------------------

adapter_dbscan <- function(experiment, method, spec) {
  k <- as.integer(spec$k %||% 20L)
  list(
    prepare = function(dat) t(as_dense(dat$X)),
    run = function(M) dbscan::sNN(M, k = k),
    canonical = function(out) {
      ## Deliberately NOT converted to a weight matrix. dbscan reports shared
      ## neighbour COUNTS, not Jaccard coefficients, so no conversion would
      ## make it numerically comparable. It is here as a workflow reference.
      out$shared
    }
  )
}

## --- dispatch ---------------------------------------------------------------

get_adapter <- function(package, experiment, method, spec) {
  if (!adapter_exists(package, experiment, method)) {
    stop("no adapter for package '", package, "', experiment '", experiment,
         "', method '", method, "'", call. = FALSE)
  }
  ad <- switch(package,
    sparseDist    = adapter_sparseDist(experiment, method, spec),
    proxyC        = adapter_proxyC(experiment, method, spec),
    text2vec      = adapter_text2vec(experiment, method, spec),
    coop          = adapter_coop(experiment, method, spec),
    parallelDist  = adapter_parallelDist(experiment, method, spec),
    philentropy   = adapter_philentropy(experiment, method, spec),
    BiocNeighbors = adapter_BiocNeighbors(experiment, method, spec),
    bluster       = adapter_bluster(experiment, method, spec),
    dbscan        = adapter_dbscan(experiment, method, spec),
    stop("unknown package '", package, "'", call. = FALSE))

  stopifnot(is.function(ad$prepare), is.function(ad$run),
            is.function(ad$canonical))
  ad
}
