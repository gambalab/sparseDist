#' @keywords internal
#'
#' @section Typical workflow:
#' \enumerate{
#'   \item \code{\link{sparseDist}} computes a full column-by-column distance
#'     or similarity matrix. Memory grows with the square of the number of
#'     columns, so this suits moderate problems.
#'   \item \code{\link{sparseKNN}} finds only the \code{k} nearest columns of
#'     each column, in blocks, so peak memory stays bounded. Use it when the
#'     full matrix would not fit.
#'   \item \code{\link{sparseSNN}} reweights that neighbour graph by
#'     neighbourhood overlap, giving a sparse adjacency matrix ready for
#'     community detection.
#' }
#'
#' @section Metrics:
#' All three functions share the same \code{method} argument:
#' \code{"binary"} (Jaccard on the sparsity pattern, the default),
#' \code{"cosine"}, \code{"euclidean"}, \code{"manhattan"}, \code{"pearson"},
#' \code{"js"} (Jensen-Shannon) and \code{"covariance"}. \code{"binary"},
#' \code{"cosine"} and \code{"pearson"} have both a distance and a similarity
#' form, selected with \code{dist}; the others have only one.
#'
#' The Jaccard, cosine, Euclidean and Manhattan kernels walk only the stored
#' non-zero entries, so their cost follows sparsity. Pearson and covariance
#' densify each column, and Jensen-Shannon takes dense input, so those are
#' better suited to matrices that are not extremely sparse.
#'
#' @section Parallelism:
#' The kernels use OpenMP where the compiler provides it. Every exported
#' function takes \code{ncores}, which defaults to 1; passing \code{0}
#' auto-detects the available cores and uses that number minus one. Results do
#' not depend on the thread count: ties are broken deterministically and each
#' thread writes to disjoint output positions.
"_PACKAGE"
