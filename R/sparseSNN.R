#' Shared-nearest-neighbour graph from a k-nearest-neighbour index
#'
#' Reweights a k-nearest-neighbour graph by neighbourhood overlap. The weight of
#' the edge between \code{i} and \code{j} is the Jaccard coefficient between the
#' two neighbour sets, \code{|N(i) intersect N(j)| / |N(i) union N(j)|}.
#'
#' @details
#' This is the second stage of graph-based clustering: build a kNN graph with
#' \code{\link{sparseKNN}}, reweight it here, then partition the result with a
#' community-detection algorithm such as those in \pkg{igraph}. Overlap
#' weighting suppresses edges between points that happen to be close but sit in
#' different neighbourhoods, which makes the community structure much clearer
#' than raw distances.
#'
#' \code{edges} selects which pairs receive a weight, and the two settings give
#' genuinely different graphs:
#'
#' \describe{
#'   \item{\code{"knn"}}{(default) a weight for every \code{j} listed among the
#'     neighbours of \code{i}. The edge set is the kNN graph itself.}
#'   \item{\code{"shared"}}{a weight for every pair sharing at least one
#'     neighbour, whether or not they are kNN-adjacent. This is the convention
#'     used by \pkg{Seurat} and by \code{bluster::neighborsToSNNGraph(type =
#'     "jaccard")}.}
#' }
#'
#' The retained OFF-DIAGONAL edges from \code{"knn"} form a subset of those
#' from \code{"shared"} -- usually a proper and substantially sparser subset,
#' though the two coincide on sufficiently small or dense neighbourhood systems
#' -- and wherever both emit an edge the weights are identical; only the edge
#' set differs. If \code{idx} explicitly contains self indices, \code{"knn"} may
#' additionally carry diagonal self-loops; \code{"shared"} emits only
#' \code{j > i} and so never produces a diagonal edge. Two points can share a
#' neighbour
#' without either appearing in the other's neighbour list, and \code{"shared"}
#' keeps such a pair while \code{"knn"} does not. \code{"shared"} is therefore
#' denser, on the order of \code{k} times as many edges, which costs both memory
#' and downstream clustering time. \code{"knn"} is cheaper and may yield sharper
#' communities; the effect on clustering should be assessed for the application.
#' Use \code{"shared"} when results must line up with a \pkg{Seurat} or
#' \pkg{bluster} pipeline, and raise \code{prune} to keep the graph manageable.
#'
#' Agreement with \code{bluster::neighborsToSNNGraph(type = "jaccard")} is
#' exact, given a \code{sparseKNN} search run with \code{include_self = FALSE}
#' and \code{include_self = TRUE} here. \pkg{Seurat} instead computes
#' \code{overlap / (2k - overlap)}, which coincides with the true union only
#' when every neighbour set holds exactly \code{k} members; results therefore
#' match for regular fixed-size neighbourhoods with aligned self conventions,
#' but can differ where rows carry padding, duplicates, or differing effective
#' sizes. The formula used here is the correct one in those cases.
#'
#' With \code{edges = "knn"} the graph is \strong{directed}, because \code{j}
#' being a neighbour of \code{i} does not make \code{i} a neighbour of \code{j}.
#' The weights themselves are symmetric -- the Jaccard coefficient does not
#' depend on direction -- so \code{w[i, j]} and \code{w[j, i]} agree wherever
#' both edges exist. Use \code{symmetrise} to obtain an undirected graph. With
#' \code{edges = "shared"} the edge set is symmetric by construction, so the
#' result is always undirected and \code{symmetrise} does not apply.
#'
#' @param idx Either the list returned by a self-search
#'   \code{\link{sparseKNN}} call, or an \code{n} by \code{k} matrix of
#'   1-based neighbour indices. Every non-missing entry must index a row of the
#'   same matrix, since it names that node's neighbourhood; \code{NA} entries
#'   are treated as padding and ignored. A cross-reference result from
#'   \code{sparseKNN(X, Y)} is rejected, because its indices refer to the
#'   columns of \code{Y} rather than to rows of \code{idx}. A list is accepted
#'   only when it still carries the \code{self_search} marker that
#'   \code{sparseKNN} attaches; subsetting a result drops that marker, so pass
#'   the result whole, or pass \code{result$idx} directly.
#' @param include_self Logical; whether a point belongs to its own neighbour
#'   set when the sets are compared. The default \code{TRUE} is the usual
#'   convention and makes a point's neighbourhood overlap with itself
#'   completely. It decides membership outright: a self index already present
#'   in a row of \code{idx} is removed when \code{FALSE}, so the result does
#'   not depend on the \code{include_self} used when the neighbours were
#'   found. Note that it changes only the sets used to compute the weights; it
#'   does not add or remove diagonal edges. A self-loop appears only when a row
#'   of \code{idx} actually lists its own index, which cannot happen when
#'   \code{edges = "shared"}.
#' @param prune Numeric in \code{[0, 1]}; edges with a weight at or below this
#'   value are dropped. The default \code{0} removes only pairs with no shared
#'   neighbours at all. Raising it thins the graph before clustering, and
#'   \code{1} yields an empty graph since the test is strict. With
#'   \code{edges = "shared"} the test is applied inside the kernel, so pruned
#'   edges are never materialised -- raising \code{prune} genuinely lowers peak
#'   memory rather than only shrinking the result.
#' @param symmetrise Logical; if \code{TRUE} (the default) the result is made
#'   undirected by keeping the larger of \code{w[i, j]} and \code{w[j, i]}.
#'   Set to \code{FALSE} to keep the directed graph. Ignored when
#'   \code{edges = "shared"}, whose edge set is symmetric by construction; a
#'   warning is issued if it is explicitly set to \code{FALSE}.
#' @param edges Either \code{"knn"} (the default) or \code{"shared"}; see
#'   Details.
#' @param ncores Number of threads. Defaults to 1.
#' @param verbose Logical; draw a progress bar while the weights are computed.
#'
#' @return A sparse \code{n} by \code{n} matrix of class \code{dgCMatrix} whose
#'   non-zero entries are the retained edge weights, carrying the row names of
#'   \code{idx} on both axes when present.
#'
#' @seealso \code{\link{sparseKNN}} to produce \code{idx}.
#'
#' @examples
#' set.seed(1)
#' X <- abs(Matrix::rsparsematrix(200, 40, density = 0.3))
#' colnames(X) <- paste0("c", seq_len(ncol(X)))
#'
#' nn  <- sparseKNN(X, k = 5, method = "binary", ncores = 1, verbose = FALSE)
#' snn <- sparseSNN(nn, ncores = 1, verbose = FALSE)
#' snn[1:5, 1:5]
#'
#' # thin the graph before clustering
#' snn <- sparseSNN(nn, prune = 1 / 15, ncores = 1, verbose = FALSE)
#'
#' # Seurat / bluster convention: every pair sharing a neighbour
#' snn_shared <- sparseSNN(nn, edges = "shared", ncores = 1, verbose = FALSE)
#' Matrix::nnzero(snn_shared) >= Matrix::nnzero(snn)
#'
#' @export
sparseSNN <- function(idx, include_self = TRUE, prune = 0,
                      symmetrise = TRUE, edges = c("knn", "shared"),
                      ncores = 1, verbose = TRUE) {

  ## Accept the sparseKNN() result directly, which is the common case.
  if (is.list(idx)) {
    ## The marker must be PRESENT and TRUE, not merely "not FALSE": ordinary
    ## subsetting drops non-name attributes, so cross[c("idx", "dist")] would
    ## otherwise arrive unmarked and be misread as a self-search -- silently
    ## producing a graph over the wrong node set. Requiring the marker turns
    ## that into an error, and anyone who really means it can pass the index
    ## matrix directly.
    ss <- attr(idx, "self_search", exact = TRUE)
    if (identical(ss, FALSE)) {
      stop("sparseSNN() requires a self-search sparseKNN() result; the ",
           "indices from sparseKNN(X, Y) refer to the columns of Y, which ",
           "are a different node set.")
    }
    if (!isTRUE(ss)) {
      stop("'idx' must be a matrix, or a list produced by sparseKNN(). This ",
           "list carries no 'self_search' marker -- it may have been dropped ",
           "by subsetting -- so it cannot be confirmed to describe a single ",
           "node set. Pass the index matrix itself if that is intended.")
    }
    if (is.null(idx$idx)) {
      stop("'idx' must be a matrix or the list returned by sparseKNN().")
    }
    idx <- idx$idx
  }
  if (!is.matrix(idx)) stop("'idx' must be a matrix or a sparseKNN() result.")

  n <- nrow(idx)
  if (n < 1L || ncol(idx) < 1L) {
    stop("'idx' must have at least one row and one column.")
  }

  ## Validate BEFORE coercing: storage.mode(idx) <- "integer" would quietly
  ## turn 1.9 into 1 and build a graph the caller never asked for. The C++
  ## range check stays as a second line of defence.
  if (!is.numeric(idx)) stop("'idx' must be a numeric or integer matrix.")
  bad <- !is.na(idx) & (!is.finite(idx) | idx != floor(idx) |
                        idx < 1 | idx > n)
  if (any(bad)) {
    stop("'idx' entries must be NA or finite, whole-number, 1-based row ",
         "indices in [1, nrow(idx)].")
  }
  storage.mode(idx) <- "integer"

  chk_flag <- function(value, name) {
    if (!is.logical(value) || length(value) != 1L || is.na(value)) {
      stop("'", name, "' must be TRUE or FALSE.")
    }
  }
  chk_flag(include_self, "include_self")
  chk_flag(symmetrise, "symmetrise")
  chk_flag(verbose, "verbose")

  edges <- match.arg(edges)
  ## symmetrise is meaningless for the shared edge set, which is symmetric by
  ## construction. Warn only when the caller asked for something we are not
  ## doing, rather than on every default-valued call.
  if (identical(edges, "shared") && !missing(symmetrise) && !symmetrise) {
    warning("'symmetrise' is ignored when edges = \"shared\": that edge set ",
            "is symmetric by construction.")
  }

  if (!is.numeric(prune) || length(prune) != 1L || is.na(prune) ||
      !is.finite(prune) || prune < 0 || prune > 1) {
    stop("'prune' must be a finite number between 0 and 1.")
  }
  if (!is.numeric(ncores) || length(ncores) != 1L || is.na(ncores) ||
      !is.finite(ncores) || ncores != floor(ncores) || ncores < 0 ||
      ncores > .Machine$integer.max) {
    stop("'ncores' must be a non-negative integer no larger than ",
         ".Machine$integer.max.")
  }
  ncores <- as.integer(ncores)
  if (ncores == 0L) ncores <- .detect_cores()

  nm <- rownames(idx)
  dn <- if (is.null(nm)) NULL else list(nm, nm)

  if (identical(edges, "shared")) {
    ## The kernel returns 1-based UPPER-TRIANGLE triplets only, with `prune`
    ## already applied -- this edge set runs to roughly n * k^2 entries, so
    ## filtering in C++ rather than here is what keeps peak memory down.
    tri <- snnJaccardShared(idx = idx, include_self = include_self,
                            prune = prune, ncores = ncores, verbose = verbose)

    ## symmetric = TRUE builds a dsCMatrix, which stores one triangle; the
    ## conversion to dgCMatrix below is what doubles it. Assembling both
    ## triangles by hand instead would need c(i, j) and c(j, i) vectors and
    ## peaks no lower, so this is the cheaper route to the documented class.
    g <- Matrix::sparseMatrix(
      i = tri$i, j = tri$j, x = tri$x,
      dims = c(n, n), dimnames = dn,
      symmetric = TRUE
    )
    return(methods::as(methods::as(g, "generalMatrix"), "CsparseMatrix"))
  }

  w <- snnJaccard(idx = idx, include_self = include_self, ncores = ncores,
                  verbose = verbose)

  ## Assemble the triplets. Every (i, t) slot is one candidate edge; NA slots
  ## are padding, and weights at or below `prune` are dropped.
  keep <- !is.na(w) & !is.na(idx) & (w > prune)

  g <- Matrix::sparseMatrix(
    i = as.vector(row(idx)[keep]),
    j = as.vector(idx[keep]),
    x = as.vector(w[keep]),
    dims = c(n, n),
    dimnames = dn,
    ## A row may list the same neighbour twice, which would otherwise emit the
    ## same (i, j) pair twice and sparseMatrix() would SUM them, producing a
    ## weight above 1. Duplicates necessarily carry the same weight here, so
    ## keeping the last is exactly right.
    use.last.ij = TRUE
  )

  if (symmetrise) {
    ## Weights are direction-free, so where both directions exist they agree
    ## and either copy is correct. pmax() has no sparse method: it densifies
    ## both operands (n^2 logicals) and, past n = 46341, overflows the integer
    ## index. Building the union from the triplets keeps everything sparse.
    gt <- Matrix::t(g)
    g  <- g + gt - gt * (g > 0)
  }

  methods::as(g, "dgCMatrix")
}
