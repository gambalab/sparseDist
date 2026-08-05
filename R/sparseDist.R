#' Optimized distance calculation on sparse matrices
#'
#' Computes column-wise distance (or similarity) matrices in parallel using
#' multiple threads. Distances are computed among the columns of a sparse data
#' matrix. The code is optimized for highly sparse, large matrices.
#'
#' @param X A matrix whose columns are variables and rows are observations.
#'   Sparse input (a \code{Matrix::dgCMatrix}, or any \code{CsparseMatrix}) is
#'   recommended; dense matrices and data frames are coerced automatically.
#' @param Y Optional second matrix with the same number of rows as \code{X}.
#'   When supplied, a rectangular (cross) matrix between the columns of \code{X}
#'   and the columns of \code{Y} is returned.
#' @param method Character; the distance to compute. One of \code{"binary"}
#'   (Jaccard on the sparsity pattern; the default), \code{"jaccard"} (an alias
#'   for \code{"binary"}), \code{"euclidean"}, \code{"manhattan"},
#'   \code{"pearson"}, \code{"js"} (Jensen-Shannon) or \code{"covariance"}.
#' @param ncores Numeric; number of threads to use. The default is \code{1}
#'   (serial). Pass \code{0} to auto-detect the available cores and use that
#'   number minus one. Note that CRAN limits package checks to two cores, so
#'   examples, tests and vignettes should pass \code{ncores = 1} or \code{2}
#'   rather than relying on auto-detection.
#' @param verbose Logical; show a progress bar. Default \code{TRUE}.
#' @param full Logical; return the full symmetric matrix (\code{TRUE}) or only
#'   the lower triangle (\code{FALSE}, the default). This applies to every
#'   single-matrix method. It is ignored when \code{Y} is supplied, in which
#'   case the full rectangular matrix is always returned.
#' @param diag Logical; compute the diagonal explicitly in the pairwise loop.
#'   Default \code{FALSE}. The diagonal of a \code{"binary"} or
#'   \code{"pearson"} result is correct either way (see \emph{Details}); for
#'   \code{"covariance"} it is the column variance and is only computed when
#'   \code{diag = TRUE}.
#' @param dist Logical; for \code{"binary"} and \code{"pearson"}, return the
#'   distance (\code{TRUE}, the default) or the similarity/coefficient
#'   (\code{FALSE}). Ignored by the other methods. Note that this also selects
#'   the storage of the result -- see \emph{Value}.
#'
#' @details
#' For \code{method = "js"} the columns must be non-negative (and should sum to
#' 1 for a genuine Jensen-Shannon divergence). Non-negativity is enforced by the
#' underlying kernel and an error is raised otherwise; normalization is left to
#' the caller. The value returned is the \emph{standard} Jensen-Shannon
#' distance \code{sqrt(JSD)} -- a true metric, bounded above by
#' \code{sqrt(log(2))} in nats. (Versions of this package before the fix
#' returned \code{sqrt(2 * JSD)}, i.e. \code{sqrt(2)} times larger.)
#'
#' In similarity mode (\code{dist = FALSE}) the diagonal of the
#' \code{"binary"} and \code{"pearson"} results is always 1, the
#' self-similarity of a column, irrespective of \code{diag}. In distance mode
#' it is 0. \code{diag} therefore controls how much work the kernel does, not
#' whether the diagonal is meaningful.
#'
#' The result is accumulated in a dense buffer inside the C++ kernels regardless
#' of the storage that is finally returned. That is required for correctness of
#' the parallel fill, not merely for speed: inserting into a compressed-sparse
#' structure from several threads would corrupt it. Peak memory is therefore
#' O(ncol^2) for every method.
#'
#' @return A matrix of distances or similarities, whose storage follows the
#'   nature of the result:
#'   \itemize{
#'     \item a \strong{dense} base \code{matrix} for distance-valued results --
#'       \code{"euclidean"}, \code{"manhattan"}, \code{"js"}, and
#'       \code{"binary"}/\code{"pearson"} with \code{dist = TRUE}. For sparse
#'       input most column pairs share nothing, which makes their \emph{distance}
#'       non-zero, so these matrices are nearly full and a sparse representation
#'       would cost more memory than it saves. A dense result also means a stored
#'       \code{0} unambiguously denotes identical columns, and a lower-triangular
#'       result can be passed straight to \code{\link[stats]{as.dist}}.
#'     \item a \strong{sparse} \code{dgCMatrix} for coefficient-valued results --
#'       \code{"covariance"}, and \code{"binary"}/\code{"pearson"} with
#'       \code{dist = FALSE}. Jaccard similarity is exactly \code{0} for disjoint
#'       column pairs, so these zeros are real. (For \code{"pearson"} and
#'       \code{"covariance"} the sparsity is only structural -- the unwritten
#'       upper triangle when \code{full = FALSE}, plus degenerate columns mapped
#'       to \code{0} -- since disjoint columns still give a small non-zero value.)
#'   }
#'   In both cases the dimension names are taken from the column names of
#'   \code{X} (rows) and of \code{Y} (columns), or of \code{X} on both axes when
#'   \code{Y} is \code{NULL}.
#'
#' @examples
#' set.seed(1)
#' X <- abs(Matrix::rsparsematrix(100, 8, density = 0.3))
#' colnames(X) <- paste0("c", seq_len(ncol(X)))
#'
#' # Jaccard distance among the columns: dense lower triangle
#' sparseDist(X, method = "binary", ncores = 1, verbose = FALSE)[1:4, 1:4]
#'
#' # the similarity form instead: sparse, with a unit diagonal
#' sparseDist(X, method = "binary", dist = FALSE, ncores = 1,
#'            verbose = FALSE)[1:4, 1:4]
#'
#' # cross-distances between the columns of two matrices
#' Y <- X[, 1:3]
#' sparseDist(X, Y, method = "euclidean", ncores = 1, verbose = FALSE)[1:4, ]
#'
#' # Jensen-Shannon requires non-negative columns; normalise them to sum to 1
#' # for a true divergence (the kernel does not normalise for you)
#' P <- as.matrix(X)
#' keep <- colSums(P) > 0
#' P <- sweep(P[, keep, drop = FALSE], 2, colSums(P[, keep, drop = FALSE]), "/")
#' sparseDist(P, method = "js", full = TRUE, diag = TRUE,
#'            ncores = 1, verbose = FALSE)[1:4, 1:4]
#'
#' @useDynLib sparseDist, .registration = TRUE
#' @import Matrix
#' @import Rcpp
#' @importFrom methods as is
#' @export
sparseDist <- function(X, Y = NULL, method = "binary", ncores = 1,
                       verbose = TRUE, full = FALSE, diag = FALSE, dist = TRUE) {

  if (is.null(X)) stop("'X' must be provided.")

  method <- match.arg(
    method,
    choices = c("binary", "jaccard", "euclidean", "manhattan",
                "pearson", "js", "covariance")
  )
  if (method == "jaccard") method <- "binary"

  if (ncores == 0) ncores <- .detect_cores()

  ## Capture names for the result, then coerce each input to the type its kernel
  ## expects: dense for "js", column-compressed sparse for every other method.
  ## This keeps X/Y consistent with the C++ argument types and lets callers pass
  ## dense matrices or data frames as well as sparse ones.
  cnX <- colnames(X)
  cnY <- if (is.null(Y)) NULL else colnames(Y)

  if (method == "js") {
    X <- as.matrix(X)
    if (!is.null(Y)) Y <- as.matrix(Y)
  } else {
    X <- .as_csparse(X)
    if (!is.null(Y)) Y <- .as_csparse(Y)
  }

  if (is.null(Y)) {
    r <- switch(method,
      binary     = fastJacc(m = X, ncores = ncores, verbose = verbose, full = full, diag = diag, dist = dist),
      euclidean  = fastEuclidean(m = X, ncores = ncores, verbose = verbose, full = full, diag = diag),
      manhattan  = fastManhattan(m = X, ncores = ncores, verbose = verbose, full = full, diag = diag),
      pearson    = fastCorr(m = X, ncores = ncores, verbose = verbose, full = full, diag = diag, dist = dist),
      js         = fastJS(m = X, ncores = ncores, verbose = verbose, full = full, diag = diag),
      covariance = fastCov(m = X, ncores = ncores, verbose = verbose, full = full, diag = diag)
    )
  } else {
    r <- switch(method,
      binary     = fastJacc2(m = X, m2 = Y, ncores = ncores, verbose = verbose, dist = dist),
      euclidean  = fastEuclidean2(m = X, m2 = Y, ncores = ncores, verbose = verbose),
      manhattan  = fastManhattan2(m = X, m2 = Y, ncores = ncores, verbose = verbose),
      pearson    = fastCorr2(m = X, m2 = Y, ncores = ncores, verbose = verbose, dist = dist),
      js         = fastJS2(m = X, m2 = Y, ncores = ncores, verbose = verbose),
      covariance = fastCov2(m = X, m2 = Y, ncores = ncores, verbose = verbose)
    )
  }

  if (is.null(Y)) {
    rownames(r) <- colnames(r) <- cnX
  } else {
    rownames(r) <- cnX
    colnames(r) <- cnY
  }

  r
}

## Coerce dense matrices, data frames, or other sparse formats to a
## column-compressed sparse matrix (dgCMatrix) -- the representation the
## RcppArmadillo kernels consume. Input that is already CsparseMatrix is
## returned unchanged (no copy).
.as_csparse <- function(M) {
  if (is.data.frame(M)) M <- as.matrix(M)
  if (methods::is(M, "CsparseMatrix")) return(M)
  methods::as(M, "CsparseMatrix")
}

## Usable thread count: all detected cores minus one (at least 1).
##
## Calls the Rcpp-exported detectCoresCpp() rather than .Call("detectCoresCpp").
## A bare string symbol is resolved by searching the loaded DLLs at run time,
## which R CMD check flags ("Found no calls to: R_registerRoutines /
## R_useDynamicSymbols", or a NOTE about non-registered native calls) and which
## can pick the wrong symbol when several packages are loaded. Functions marked
## // [[Rcpp::export]] get a registered R-level wrapper generated for them, so
## calling that wrapper is both safe and check-clean.
##
## Requires the package NAMESPACE to contain
##     useDynLib(sparseDist, .registration = TRUE)
## which the @useDynLib roxygen tag on sparseDist() above generates. Without it
## the shared object is never attached to the namespace and every call fails
## with '"_sparseDist_detectCoresCpp" not available for .Call()'.
## Renamed from detectCores() so it no longer masks parallel::detectCores();
## if the old name was exported, add an alias in NAMESPACE.
.detect_cores <- function() {
  n <- as.integer(detectCoresCpp())
  if (is.na(n) || n < 2L) 1L else n - 1L
}
