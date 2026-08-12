## sparseKNNCross() has an exact oracle: each direction must equal the
## sparseKNN() call it replaces, for every input and every block_size. These
## tests assert identical(), not all.equal() -- the same slab values are reduced
## by comparators of the same form, so any difference is a bug in the
## accumulation, not floating-point noise.

as_dgc <- function(M) {
  M <- as(as(as(M, "dMatrix"), "generalMatrix"), "CsparseMatrix")
  M
}
mk <- function(nr, nc, dens, seed) {
  set.seed(seed)
  M <- as_dgc(abs(Matrix::rsparsematrix(nr, nc, density = dens, repr = "C")))
  M
}
mk_bin <- function(nr, nc, dens, seed) { M <- mk(nr, nc, dens, seed); M@x[] <- 1; M }

expect_matches_sparseKNN <- function(X, Y, k, block_size, method, dist = TRUE,
                                     drop_degenerate = TRUE) {
  lab <- sprintf("%s dist=%s bs=%d k=%d dims=%dx%d/%dx%d", method, dist,
                 block_size, k, nrow(X), ncol(X), nrow(Y), ncol(Y))
  cr <- suppressWarnings(sparseKNNCross(X, Y, k = k, method = method, dist = dist,
                                        drop_degenerate = drop_degenerate,
                                        block_size = block_size, ncores = 2L,
                                        verbose = FALSE))
  fw <- suppressWarnings(sparseKNN(X, Y, k = k, method = method, dist = dist,
                                   drop_degenerate = drop_degenerate,
                                   block_size = block_size, ncores = 2L,
                                   verbose = FALSE))
  rv <- suppressWarnings(sparseKNN(Y, X, k = k, method = method, dist = dist,
                                   drop_degenerate = drop_degenerate,
                                   block_size = block_size, ncores = 2L,
                                   verbose = FALSE))
  testthat::expect_identical(cr$forward, fw, info = paste("forward:", lab))
  testthat::expect_identical(cr$reverse, rv, info = paste("reverse:", lab))
}

testthat::test_that("both directions match the equivalent sparseKNN calls", {
  X <- mk_bin(300, 80, 0.05, 1)
  Y <- mk_bin(300, 50, 0.05, 2)
  for (m in c("binary", "cosine", "euclidean", "manhattan")) {
    expect_matches_sparseKNN(X, Y, k = 7L, block_size = 16L, method = m)
  }
  ## Similarity form flips the ranking direction; check it is threaded through.
  for (m in c("binary", "cosine")) {
    expect_matches_sparseKNN(X, Y, k = 7L, block_size = 16L, method = m, dist = FALSE)
  }
})

testthat::test_that("the coefficient and dense methods match too", {
  ## The reverse direction reduces sparseDist(Y, Xblock), whereas its oracle
  ## sparseKNN(Y, X) reduces sparseDist(X, Yblock). Exact equality therefore
  ## additionally requires kernel(Y_i, X_j) == kernel(X_j, Y_i) BIT-FOR-BIT,
  ## not merely mathematically. That is an assumption about the summation order
  ## inside fastCorr2/fastCov2/fastJS2, so it belongs in the suite rather than
  ## in a reviewer's scratch experiment -- these are the methods where a
  ## reassociated accumulation would be most likely to break it.
  X <- mk(300, 40, 0.10, 21)
  Y <- mk(300, 25, 0.10, 22)
  for (bs in c(1L, 7L, 40L)) {
    expect_matches_sparseKNN(X, Y, k = 6L, block_size = bs, method = "pearson", dist = TRUE)
    expect_matches_sparseKNN(X, Y, k = 6L, block_size = bs, method = "pearson", dist = FALSE)
    expect_matches_sparseKNN(X, Y, k = 6L, block_size = bs, method = "covariance")
    expect_matches_sparseKNN(X, Y, k = 6L, block_size = bs, method = "js")
  }
})

testthat::test_that("the running state is independent of block_size", {
  X <- mk_bin(200, 61, 0.06, 3)     # 61 is prime: every block_size leaves a remainder
  Y <- mk_bin(200, 37, 0.06, 4)
  ref <- sparseKNNCross(X, Y, k = 5L, method = "binary", block_size = 61L,
                        ncores = 1L, verbose = FALSE)
  for (bs in c(1L, 2L, 7L, 13L, 60L, 61L, 62L, 1000L)) {
    got <- sparseKNNCross(X, Y, k = 5L, method = "binary", block_size = bs,
                          ncores = 1L, verbose = FALSE)
    testthat::expect_identical(got, ref, info = paste("block_size", bs))
  }
})

testthat::test_that("results do not depend on thread count", {
  X <- mk_bin(200, 90, 0.05, 5)
  Y <- mk_bin(200, 70, 0.05, 6)
  one <- sparseKNNCross(X, Y, k = 9L, method = "binary", block_size = 11L,
                        ncores = 1L, verbose = FALSE)
  for (nc in c(2L, 4L, 8L)) {
    testthat::expect_identical(
      sparseKNNCross(X, Y, k = 9L, method = "binary", block_size = 11L,
                     ncores = nc, verbose = FALSE),
      one, info = paste("ncores", nc))
  }
})

testthat::test_that("massive ties are broken by the lower global index", {
  ## A binary matrix whose columns are all identical makes every pairwise
  ## Jaccard equal, so the entire result is decided by tie-breaking. This is
  ## the case most likely to expose a >= where a > was meant.
  X <- as_dgc(Matrix::sparseMatrix(i = rep(1:20, 40), j = rep(1:40, each = 20),
                                   x = 1, dims = c(50, 40)))
  Y <- as_dgc(Matrix::sparseMatrix(i = rep(1:20, 25), j = rep(1:25, each = 20),
                                   x = 1, dims = c(50, 25)))
  for (bs in c(1L, 3L, 8L, 40L)) {
    expect_matches_sparseKNN(X, Y, k = 6L, block_size = bs, method = "binary")
  }
})

testthat::test_that("degenerate columns are excluded on both axes", {
  X <- mk_bin(150, 40, 0.06, 7); X[, c(1, 5, 40)] <- 0; X <- as_dgc(Matrix::drop0(X))
  Y <- mk_bin(150, 30, 0.06, 8); Y[, c(2, 30)]     <- 0; Y <- as_dgc(Matrix::drop0(Y))
  for (dd in c(TRUE, FALSE)) {
    ## drop_degenerate = FALSE is only well-defined for the distance methods
    ## that treat an empty column as a legitimate point at the origin.
    meths <- if (dd) c("binary", "cosine", "euclidean") else c("euclidean", "manhattan")
    for (m in meths) {
      expect_matches_sparseKNN(X, Y, k = 4L, block_size = 7L, method = m,
                               drop_degenerate = dd)
    }
  }
})

testthat::test_that("clamping counts ELIGIBLE candidates, not raw matrix width", {
  ## Both sides are 20 columns wide, so any implementation that clamped on
  ## ncol() would pass. Only 14 columns of X and 3 of Y are non-degenerate, so
  ## with k = 10 the forward direction must truncate to 3 while the reverse one
  ## keeps all 10 -- widths alone cannot distinguish these.
  X <- mk_bin(120, 20, 0.10, 31); X[, 15:20] <- 0; X <- as_dgc(Matrix::drop0(X))
  Y <- mk_bin(120, 20, 0.10, 32); Y[,  4:20] <- 0; Y <- as_dgc(Matrix::drop0(Y))
  testthat::expect_identical(sum(Matrix::colSums(abs(X)) > 0), 14L)
  testthat::expect_identical(sum(Matrix::colSums(abs(Y)) > 0), 3L)
  for (bs in c(1L, 6L, 20L)) {
    expect_matches_sparseKNN(X, Y, k = 10L, block_size = bs, method = "binary")
    expect_matches_sparseKNN(Y, X, k = 10L, block_size = bs, method = "binary")
  }
})

testthat::test_that("k larger than one side's candidate pool is clamped per direction", {
  ## Y has fewer columns than k, so the forward direction must truncate while
  ## the reverse one does not -- a single shared k would be wrong here.
  X <- mk_bin(120, 30, 0.08, 9)
  Y <- mk_bin(120,  4, 0.08, 10)
  expect_matches_sparseKNN(X, Y, k = 10L, block_size = 8L, method = "binary")
  expect_matches_sparseKNN(Y, X, k = 10L, block_size = 8L, method = "binary")
})

testthat::test_that("single-column inputs work on either side", {
  X <- mk_bin(80, 1, 0.2, 11)
  Y <- mk_bin(80, 12, 0.1, 12)
  expect_matches_sparseKNN(X, Y, k = 3L, block_size = 5L, method = "binary")
  expect_matches_sparseKNN(Y, X, k = 1L, block_size = 5L, method = "binary")
})

testthat::test_that("the jaccard alias is identical to binary", {
  ## "jaccard" is an advertised eighth method string that the implementation
  ## folds into "binary" immediately. Cheap now, but it is exactly the kind of
  ## aliasing that a later .knn_setup() extraction could drop on the floor.
  X <- mk_bin(100, 18, 0.1, 51)
  Y <- mk_bin(100, 11, 0.1, 52)
  for (dst in c(TRUE, FALSE)) {
    testthat::expect_identical(
      sparseKNNCross(X, Y, k = 4L, method = "jaccard", dist = dst,
                     block_size = 5L, ncores = 2L, verbose = FALSE),
      sparseKNNCross(X, Y, k = 4L, method = "binary", dist = dst,
                     block_size = 5L, ncores = 2L, verbose = FALSE))
  }
})

testthat::test_that("a single matrix is rejected rather than silently mishandled", {
  X <- mk_bin(60, 10, 0.1, 13)
  testthat::expect_error(sparseKNNCross(X, k = 3L, verbose = FALSE), "required")
  testthat::expect_error(sparseKNNCross(X, NULL, k = 3L, verbose = FALSE), "required")
  testthat::expect_error(
    sparseKNNCross(X, mk_bin(59, 10, 0.1, 14), k = 3L, verbose = FALSE),
    "same number of rows")
})

testthat::test_that("randomised configurations all agree with sparseKNN", {
  ## Hand-picked cases cover the shapes I thought of; this covers the ones I
  ## did not. Cheap enough to run on every check.
  set.seed(20240612)
  for (trial in seq_len(120)) {
    nrw <- sample(40:160, 1)
    ncx <- sample(1:45, 1)
    ncy <- sample(1:45, 1)
    dens <- runif(1, 0.02, 0.25)
    k    <- sample(1:8, 1)
    bs   <- sample(c(1:6, 7, 11, 16, 64), 1)
    meth <- sample(c("binary", "cosine", "euclidean", "manhattan",
                     "pearson", "covariance", "js"), 1)
    dst  <- if (meth %in% c("binary", "cosine", "pearson")) sample(c(TRUE, FALSE), 1) else TRUE

    X <- mk(nrw, ncx, dens, trial * 2L)
    Y <- mk(nrw, ncy, dens, trial * 2L + 1L)
    if (meth == "binary") { X@x[] <- 1; Y@x[] <- 1 }

    expect_matches_sparseKNN(X, Y, k = k, block_size = bs, method = meth, dist = dst)
  }
})

testthat::test_that("argument validation matches sparseKNN rather than coercing", {
  ## An earlier draft used as.integer()/isTRUE() here, which silently truncated
  ## fractional counts and -- via isTRUE(1) == FALSE while Rcpp coerces 1 to
  ## TRUE -- let the ranking direction disagree with the native call.
  ##
  ## Every assertion below matches the EXPECTED MESSAGE. A bare expect_error()
  ## would be satisfied by any failure at all: an earlier version of this test
  ## passed all fifteen cases while the validators were not even installed,
  ## because "could not find function" is also an error. A regression test that
  ## cannot distinguish the bug it guards from an unrelated crash is worse than
  ## none, since it reports green either way.
  X <- mk_bin(80, 12, 0.1, 41)
  Y <- mk_bin(80,  9, 0.1, 42)
  bad <- function(pattern, ...) testthat::expect_error(
    sparseKNNCross(X, Y, verbose = FALSE, ...), pattern)

  pos <- "'k' must be a positive integer"
  bad(pos, k = 2.5)             # truncation
  bad(pos, k = 0)               # below the lower bound
  bad(pos, k = Inf)             # non-finite
  bad(pos, k = NA_integer_)
  bad(pos, k = 3e9)             # above INT_MAX
  bad(pos, k = c(2L, 3L))       # not length 1

  bad("'block_size' must be a positive integer", block_size = 4.5)
  bad("'block_size' must be a positive integer", block_size = 0)

  nn <- "'ncores' must be a non-negative integer"
  bad(nn, ncores = 1.7)
  bad(nn, ncores = -1L)

  ## dist = 1 is the important one: isTRUE(1) is FALSE so the R side would rank
  ## as a similarity, while Rcpp coerces 1 to TRUE in the native call.
  bad("'dist' must be TRUE or FALSE", dist = 1)
  bad("'dist' must be TRUE or FALSE", dist = NA)
  bad("'dist' must be TRUE or FALSE", dist = c(TRUE, TRUE))
  bad("'drop_degenerate' must be TRUE or FALSE", drop_degenerate = 1)
  testthat::expect_error(sparseKNNCross(X, Y, verbose = 1),
                         "'verbose' must be TRUE or FALSE")

  ## ncores = 0 means auto-detect, not an error.
  testthat::expect_silent(
    sparseKNNCross(X, Y, k = 3L, ncores = 0L, verbose = FALSE))
})
