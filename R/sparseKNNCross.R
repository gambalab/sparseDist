#' Bidirectional k-nearest-neighbour search from a single pass
#'
#' Computes the neighbours of every column of `X` among the columns of `Y` AND
#' the neighbours of every column of `Y` among the columns of `X`, from one set
#' of distance blocks.
#'
#' The slab for a block of query columns already contains every pairwise value
#' needed for both directions -- `sparseKNN(X, Y)` reduces it down its columns
#' and throws the rest away, and a second call to `sparseKNN(Y, X)` recomputes
#' the identical numbers to reduce them across its rows. This function reduces
#' in both directions from the same slab, so the distance work is done once.
#'
#' In a sharded all-pairs search this halves the grid: shard pairs (a, b) and
#' (b, a) collapse into one task. The reverse direction cannot be finalised
#' block by block -- a reference column has been compared only against the
#' queries seen so far -- so it is accumulated in a running state by
#' `topKRowsAccum()`.
#'
#' @param X,Y Sparse column matrices with the same number of rows. Unlike
#'   [sparseKNN()], `Y` is required: with one matrix the two directions coincide
#'   and [sparseKNN()] already exploits the symmetry through `full`/`diag`.
#' @param k Neighbours to return. Truncated INDEPENDENTLY per direction if it
#'   exceeds the candidates available on that side, matching what the
#'   corresponding [sparseKNN()] call would do.
#' @param method,dist,drop_degenerate,block_size,ncores,verbose As in
#'   [sparseKNN()].
#'
#' @return A list of two elements, `forward` and `reverse`, each a list of `idx`
#'   and `dist` matrices with the same layout and `self_search` attribute as a
#'   [sparseKNN()] result. `forward` is indexed by the columns of `X` and points
#'   into `Y`; `reverse` is indexed by the columns of `Y` and points into `X`.
#'
#' @section Equivalence:
#' By construction, for any inputs and any `block_size`:
#' \preformatted{
#'   identical(sparseKNNCross(X, Y, k)$forward, sparseKNN(X, Y, k))
#'   identical(sparseKNNCross(X, Y, k)$reverse, sparseKNN(Y, X, k))
#' }
#' This is the definition the test suite enforces; treat any deviation as a bug
#' in this function rather than as a tolerable difference.
#'
#' @seealso [sparseKNN()]
#' @export
sparseKNNCross <- function(X, Y, k = 10L,
                           method = c("binary", "jaccard", "cosine", "euclidean",
                                      "manhattan", "pearson", "js", "covariance"),
                           dist = TRUE, drop_degenerate = TRUE,
                           block_size = 256L, ncores = 1L, verbose = TRUE) {

  ## ------------------------------------------------------------------------
  ## Preamble. This mirrors sparseKNN() deliberately and must be kept in step
  ## with it; the two share no code today only because factoring the setup out
  ## would touch a function that is already covered by tests. A follow-up
  ## refactor into a common .knn_setup() helper is the right end state.
  ## ------------------------------------------------------------------------
  if (missing(Y) || is.null(Y)) {
    stop("'Y' is required. For a single matrix use sparseKNN(), which already ",
         "exploits symmetry via the full/diag arguments.")
  }

  ## Strict validation, matching sparseKNN(). An earlier draft used
  ## as.integer()/isTRUE() here and thereby reintroduced exactly the bugs that
  ## validation was written to prevent: block_size = 4.5 truncating silently,
  ## ncores = Inf becoming NA, and worst of all dist = 1, where isTRUE(1) is
  ## FALSE while Rcpp coerces 1 to TRUE -- so the ranking direction and the
  ## native call would disagree. Kept as calls to the shared helpers rather
  ## than as another local copy.
  .chk_flag(dist, "dist")
  .chk_flag(drop_degenerate, "drop_degenerate")
  .chk_flag(verbose, "verbose")

  ncores     <- .chk_count(ncores, "ncores", allow_zero = TRUE)
  k          <- .chk_count(k, "k")
  block_size <- .chk_count(block_size, "block_size")
  if (ncores == 0L) ncores <- .detect_cores()

  method <- match.arg(method)
  if (method == "jaccard") method <- "binary"

  if (method %in% c("euclidean", "manhattan", "js") && !isTRUE(dist)) {
    stop("'dist = FALSE' is only meaningful for methods \"binary\", ",
         "\"cosine\" and \"pearson\"; ", method, " has no similarity form.")
  }
  decreasing <- switch(
    method,
    binary     = !isTRUE(dist),
    cosine     = !isTRUE(dist),
    pearson    = !isTRUE(dist),
    covariance = TRUE,
    euclidean  = FALSE,
    manhattan  = FALSE,
    js         = FALSE
  )

  if (nrow(X) != nrow(Y)) stop("'X' and 'Y' must have the same number of rows.")
  nq <- ncol(X); nr <- ncol(Y)
  if (is.null(nq) || nq < 1L) stop("'X' must have at least one column.")
  if (is.null(nr) || nr < 1L) stop("'Y' must have at least one column.")
  nmX <- colnames(X); nmY <- colnames(Y)

  if (method == "js") {
    X <- as.matrix(X); Y <- as.matrix(Y)
  } else {
    X <- .as_csparse(X); Y <- .as_csparse(Y)
  }

  col_is_empty <- function(M) {
    tot <- as.numeric(Matrix::colSums(abs(M)))
    !is.na(tot) & tot == 0
  }
  deg_query <- if (drop_degenerate) col_is_empty(X) else rep(FALSE, nq)
  deg_ref   <- if (drop_degenerate) col_is_empty(Y) else rep(FALSE, nr)

  ## Each direction truncates against its OWN candidate pool. Truncating both
  ## to a single k would diverge from sparseKNN() whenever the two sides differ
  ## in width, which is the normal case for the final shard of a grid.
  clamp <- function(k, avail, side) {
    if (avail < 1L) stop("No candidate neighbours available for the ", side, " direction.")
    if (k > avail) {
      warning("k (", k, ") exceeds the ", avail, " available neighbours for the ",
              side, " direction; returning ", avail, ".")
      return(avail)
    }
    k
  }
  k_fwd <- clamp(k, sum(!deg_ref),   "forward")
  k_rev <- clamp(k, sum(!deg_query), "reverse")

  ## ------------------------------------------------------------------------
  ## Result storage. The reverse state is carried across blocks and mutated in
  ## place by topKRowsAccum(); allocate it here and never alias it.
  ## ------------------------------------------------------------------------
  fwd_idx <- matrix(NA_integer_, nrow = nq, ncol = k_fwd)
  fwd_dst <- matrix(NA_real_,    nrow = nq, ncol = k_fwd)
  rev_idx <- matrix(NA_integer_, nrow = nr, ncol = k_rev)
  rev_dst <- matrix(NA_real_,    nrow = nr, ncol = k_rev)

  starts <- seq.int(1L, nq, by = block_size)
  pb <- NULL
  if (verbose) {
    pb <- utils::txtProgressBar(min = 0, max = length(starts), style = 3)
    on.exit(close(pb), add = TRUE)
  }

  ## No row of D is ever a query column's own self-comparison here: X and Y are
  ## required to be different matrices. The parameters are still passed
  ## explicitly rather than omitted, so that the kernels behave identically to
  ## the sparseKNN() path they are checked against.
  no_self_rows <- rep(0L, nr)

  for (bi in seq_along(starts)) {
    from <- starts[bi]
    to   <- min(from + block_size - 1L, nq)
    cols <- from:to

    D <- sparseDist(X = Y, Y = X[, cols, drop = FALSE], method = method,
                    ncores = ncores, verbose = FALSE, dist = dist)
    D <- as.matrix(D)

    ## Mask degenerate columns on BOTH axes, before either reduction. The
    ## forward path in sparseKNN() masks degenerate references as rows and
    ## blanks degenerate queries afterwards; doing both here as NA is
    ## equivalent for the forward direction and REQUIRED for the reverse one,
    ## where a degenerate query column is a candidate neighbour rather than a
    ## query. topKBlock and topKRowsAccum both skip non-finite values.
    if (any(deg_ref)) D[deg_ref, ] <- NA_real_
    deg_blk <- deg_query[cols]
    if (any(deg_blk)) D[, deg_blk] <- NA_real_

    ## Forward: k best rows of D per column. Final for this block.
    part <- topKBlock(D = D, k = k_fwd, decreasing = decreasing,
                      self_row = rep(0L, length(cols)), ncores = ncores)
    fwd_idx[cols, ] <- part$idx
    fwd_dst[cols, ] <- part$dist

    ## Reverse: k best columns of D per row, folded into the running state.
    ## col_offset is 0-based, so indices land in the global query numbering.
    topKRowsAccum(D = D, k = k_rev, decreasing = decreasing,
                  best_idx = rev_idx, best_val = rev_dst,
                  col_offset = from - 1L, self_col = no_self_rows,
                  ncores = ncores)

    if (verbose) utils::setTxtProgressBar(pb, bi)
  }

  ## A degenerate column gets no neighbours of its own in either direction.
  if (any(deg_query)) { fwd_idx[deg_query, ] <- NA_integer_; fwd_dst[deg_query, ] <- NA_real_ }
  if (any(deg_ref))   { rev_idx[deg_ref, ]   <- NA_integer_; rev_dst[deg_ref, ]   <- NA_real_ }

  rownames(fwd_idx) <- nmX; rownames(fwd_dst) <- nmX
  rownames(rev_idx) <- nmY; rownames(rev_dst) <- nmY

  fwd <- list(idx = fwd_idx, dist = fwd_dst)
  rev <- list(idx = rev_idx, dist = rev_dst)
  attr(fwd, "self_search") <- FALSE
  attr(rev, "self_search") <- FALSE

  list(forward = fwd, reverse = rev)
}
