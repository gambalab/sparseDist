#' Blocked k-nearest-neighbour search over the columns of a sparse matrix
#'
#' Finds, for each column of \code{X}, the \code{k} closest (or most similar)
#' columns, without ever materialising the full column-by-column matrix.
#'
#' @details
#' \code{sparseDist()} returns the whole \code{ncol(X)} by \code{ncol(X)}
#' result, so its memory grows with the square of the number of columns: at
#' 50,000 columns that is roughly 20 GB. \code{sparseKNN()} instead walks the
#' columns of \code{X} in blocks, computes one
#' \code{ncol(reference)} by \code{block_size} slab at a time with the same
#' kernels, and immediately reduces each slab to its \code{k} best entries.
#' Peak memory is therefore proportional to
#' \code{ncol(reference) * block_size} rather than \code{ncol(X)^2}, which is
#' what makes large problems feasible.
#'
#' The blocking is a pure win for memory but not for arithmetic: every block
#' needs distances to all reference columns, so the symmetric case computes
#' each pair twice rather than exploiting the triangle. That trade is
#' deliberate -- it keeps the existing, well-tested kernels untouched and
#' avoids merging partial results across threads.
#'
#' The neighbour relation is \strong{directed}. \code{j} being among the
#' neighbours of \code{i} does not imply the reverse, so the result describes a
#' directed graph. Symmetrise it yourself if an undirected graph is wanted.
#'
#' Ties at the \code{k}-th position are broken by the lower column index, so
#' results are reproducible and do not depend on \code{ncores}.
#'
#' When \code{include_self = TRUE} the exact self-comparison is restored after
#' each block: a column is at distance 0 (similarity 1) from itself even where
#' the rectangular kernel could not know that two columns are the same, such as
#' a constant column under \code{"pearson"}. The one exception is
#' \code{"covariance"}, whose self-value is the column variance.
#'
#' @param X A matrix whose columns are the query points. Sparse input
#'   (\code{dgCMatrix}) is recommended; dense matrices and data frames are
#'   coerced automatically.
#' @param Y Optional reference matrix with the same number of rows as \code{X}.
#'   Neighbours are drawn from the columns of \code{Y} when supplied, and from
#'   the columns of \code{X} itself otherwise.
#' @param k Number of neighbours to return per column.
#' @param method Distance or similarity to use; see \code{\link{sparseDist}}.
#' @param dist Logical; for \code{"binary"}, \code{"cosine"} and
#'   \code{"pearson"}, \code{TRUE}
#'   (the default) ranks by distance and keeps the \code{k} smallest values,
#'   while \code{FALSE} ranks by similarity and keeps the \code{k} largest.
#'   The other methods have only one form, so the ranking direction is fixed:
#'   \code{"euclidean"}, \code{"manhattan"} and \code{"js"} are distances
#'   (smallest kept) and reject \code{dist = FALSE}; \code{"covariance"} is a
#'   coefficient and always keeps the largest values, ignoring \code{dist}.
#'   \code{"binary"}, \code{"cosine"} and \code{"pearson"} have both forms.
#' @param include_self Logical; when \code{Y} is \code{NULL}, whether a column
#'   may be returned as its own neighbour. Defaults to \code{FALSE}, matching
#'   the usual k-nearest-neighbour convention. Ignored when \code{Y} is given,
#'   since the two matrices are then treated as unrelated.
#' @param block_size Number of query columns per block. Larger values do more
#'   work per kernel call (slightly faster) at proportionally higher peak
#'   memory. The default of 256 keeps a block near
#'   \code{ncol(reference) * 256 * 8} bytes.
#' @param ncores Number of threads, used for both the distance blocks and the
#'   top-k reduction. Defaults to 1; \code{0} auto-detects as in
#'   \code{\link{sparseDist}}. CRAN limits checks to two cores, so keep
#'   examples and tests at 1 or 2.
#' @param verbose Logical; draw a progress bar. One bar spans the whole
#'   search and advances once per block, so its resolution is set by
#'   \code{block_size}.
#'
#' @return A list with two \code{ncol(X)} by \code{k} matrices:
#'   \describe{
#'     \item{\code{idx}}{integer, 1-based column indices into the reference
#'       (\code{Y} if supplied, otherwise \code{X}), best neighbour first.}
#'     \item{\code{dist}}{the corresponding distance or similarity values.}
#'   }
#'   Both carry the column names of \code{X} as row names. When a column has
#'   fewer than \code{k} valid neighbours the remaining slots are \code{NA}.
#'   This is the layout used by \pkg{FNN} and \pkg{RANN}; a sparse adjacency
#'   matrix can be built from it directly, for example with
#'   \code{Matrix::sparseMatrix(i = row(res$idx), j = res$idx, x = res$dist)}.
#'
#' @seealso \code{\link{sparseDist}} for the full matrix.
#'
#' @examples
#' set.seed(1)
#' X <- abs(Matrix::rsparsematrix(200, 40, density = 0.3))
#' colnames(X) <- paste0("c", seq_len(ncol(X)))
#'
#' nn <- sparseKNN(X, k = 5, method = "binary", ncores = 1, verbose = FALSE)
#' head(nn$idx)
#' head(nn$dist)
#'
#' # the k most similar columns instead of the k closest
#' nn <- sparseKNN(X, k = 5, method = "binary", dist = FALSE,
#'                 ncores = 1, verbose = FALSE)
#' head(nn$idx)
#'
#' @export
sparseKNN <- function(X, Y = NULL, k = 10, method = "binary", dist = TRUE,
                      include_self = FALSE, block_size = 256L,
                      ncores = 1, verbose = TRUE) {

  if (is.null(X)) stop("'X' must be provided.")

  ## Scalar-logical validation. `dist` in particular is used twice -- once to
  ## pick the ranking direction here, once passed on to sparseDist() -- and a
  ## value such as 1 or NA would be interpreted differently by the two
  ## (isTRUE(1) is FALSE, but Rcpp coerces 1 to TRUE), silently ranking a
  ## distance as if it were a similarity.
  chk_flag <- function(value, name) {
    if (!is.logical(value) || length(value) != 1L || is.na(value)) {
      stop("'", name, "' must be TRUE or FALSE.")
    }
  }
  chk_flag(dist, "dist")
  chk_flag(include_self, "include_self")
  chk_flag(verbose, "verbose")

  ## Count-like arguments: reject anything that is not a single finite whole
  ## number rather than letting as.integer() truncate silently (block_size =
  ## 4.5), or turn a value into NA because it is non-finite (ncores = Inf) or
  ## above INT_MAX (k = 3e9), either of which would surface as an unrelated
  ## error further down.
  chk_count <- function(value, name, allow_zero = FALSE) {
    lower <- if (allow_zero) 0 else 1
    if (!is.numeric(value) || length(value) != 1L || is.na(value) ||
        !is.finite(value) || value != floor(value) || value < lower ||
        value > .Machine$integer.max) {
      stop("'", name, "' must be a ",
           if (allow_zero) "non-negative" else "positive",
           " integer no larger than .Machine$integer.max.")
    }
    as.integer(value)
  }
  ncores     <- chk_count(ncores, "ncores", allow_zero = TRUE)
  k          <- chk_count(k, "k")
  block_size <- chk_count(block_size, "block_size")

  ## Resolve ncores ONCE, so the distance slab and the top-k reduction use the
  ## same thread count (topKBlock treats 0 as 1, sparseDist treats 0 as
  ## auto-detect, which would otherwise leave the reduction serial).
  if (ncores == 0L) ncores <- .detect_cores()

  method <- match.arg(
    method,
    choices = c("binary", "jaccard", "cosine", "euclidean", "manhattan",
                "pearson", "js", "covariance")
  )
  if (method == "jaccard") method <- "binary"

  ## Ranking direction is a property of the METHOD, not of `dist` alone:
  ## sparseDist() ignores `dist` for euclidean, manhattan, js and covariance,
  ## so deriving the direction from `dist` would silently invert the ranking
  ## (e.g. returning the FARTHEST columns for method = "euclidean",
  ## dist = FALSE, or the most negative covariances by default).
  if (method %in% c("euclidean", "manhattan", "js") && !isTRUE(dist)) {
    stop("'dist = FALSE' is only meaningful for methods \"binary\", ",
         "\"cosine\" and \"pearson\"; ", method, " has no similarity form.")
  }
  decreasing <- switch(
    method,
    binary     = !isTRUE(dist),   # similarity -> keep the largest
    cosine     = !isTRUE(dist),
    pearson    = !isTRUE(dist),
    covariance = TRUE,            # always a coefficient: keep the largest
    euclidean  = FALSE,           # always a distance: keep the smallest
    manhattan  = FALSE,
    js         = FALSE
  )

  nmX <- colnames(X)
  nq  <- ncol(X)
  if (is.null(nq) || nq < 1L) stop("'X' must have at least one column.")

  selfmode <- is.null(Y)
  if (!selfmode && nrow(X) != nrow(Y)) {
    stop("'X' and 'Y' must have the same number of rows.")
  }

  ## Coerce ONCE, not once per block: sparseDist() would otherwise re-convert
  ## dense matrices, data frames and foreign sparse formats on every call.
  if (method == "js") {
    X <- as.matrix(X)
    if (!selfmode) Y <- as.matrix(Y)
  } else {
    X <- .as_csparse(X)
    if (!selfmode) Y <- .as_csparse(Y)
  }

  ref  <- if (selfmode) X else Y
  nref <- ncol(ref)

  ## Largest possible neighbour count: in self mode a column cannot be its own
  ## neighbour unless include_self is TRUE.
  avail <- if (selfmode && !include_self) nref - 1L else nref
  if (avail < 1L) stop("No candidate neighbours available.")
  if (k > avail) {
    warning("k (", k, ") exceeds the ", avail,
            " available neighbours; returning ", avail, ".")
    k <- avail
  }

  starts <- seq.int(1L, nq, by = block_size)
  idx  <- matrix(NA_integer_, nrow = nq, ncol = k)
  dst  <- matrix(NA_real_,    nrow = nq, ncol = k)

  ## One bar across the whole search, advanced per block. The inner
  ## sparseDist() calls stay silent: letting each block draw its own
  ## RcppProgress bar would restart the display length(starts) times. The
  ## kernels still honour interrupts inside a block, since their Progress
  ## object monitors the abort flag whether or not it displays anything.
  pb <- NULL
  if (verbose) {
    pb <- utils::txtProgressBar(min = 0, max = length(starts), style = 3)
    on.exit(close(pb), add = TRUE)
  }

  for (bi in seq_along(starts)) {
    from <- starts[bi]
    to   <- min(from + block_size - 1L, nq)
    cols <- from:to

    ## One slab: reference columns x this block of query columns. The full
    ## reference goes in as 'X' because sparseDist() parallelises over its
    ## first argument, which is the larger dimension here.
    D <- sparseDist(X = ref, Y = X[, cols, drop = FALSE], method = method,
                    ncores = ncores, verbose = FALSE, dist = dist)
    D <- as.matrix(D)             # coefficient methods return a sparse slab

    ## Row of D holding each query column's self-comparison, or 0 for none.
    self_row <- if (selfmode && !include_self) as.integer(cols) else rep(0L, length(cols))

    ## The rectangular kernels cannot know that a reference column and a query
    ## column are the same original column, so they lose the exact
    ## self-comparison that the single-matrix kernels special-case: a constant
    ## column's self-correlation becomes 0 (undefined -> 0), and a JS
    ## self-distance becomes sqrt(round-off) rather than 0. Restore the known
    ## values at the self positions. Only needed when the self-comparison is
    ## actually kept; otherwise those entries are excluded anyway.
    if (selfmode && include_self) {
      self_pos <- cbind(cols, seq_along(cols))
      if (method %in% c("binary", "cosine", "pearson")) {
        D[self_pos] <- if (dist) 0 else 1
      } else if (method %in% c("euclidean", "manhattan", "js")) {
        D[self_pos] <- 0
      }
      ## covariance is deliberately left alone: a column's self-covariance is
      ## its variance, which fastCov2() already computes correctly.
    }

    part <- topKBlock(D = D, k = k, decreasing = decreasing,
                      self_row = self_row, ncores = ncores)

    idx[cols, ] <- part$idx
    dst[cols, ] <- part$dist

    if (verbose) utils::setTxtProgressBar(pb, bi)
  }

  rownames(idx) <- nmX
  rownames(dst) <- nmX

  ## Flag whether the indices refer to the columns of X itself. sparseSNN()
  ## needs one node set, so it must reject a cross-reference result whose
  ## indices point into Y -- which would otherwise be silently misread as rows
  ## of the same matrix whenever ncol(X) == ncol(Y).
  res <- list(idx = idx, dist = dst)
  attr(res, "self_search") <- selfmode
  res
}
