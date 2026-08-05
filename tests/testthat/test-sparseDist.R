# Regression tests. Every block below pins down a bug that was fixed during
# review, so a future change that reintroduces one fails loudly here.
#
# All calls pass ncores = 1: CRAN allows a check to use at most two cores.

make_X <- function(n = 60, p = 8, density = 0.3, seed = 1) {
  set.seed(seed)
  X <- abs(Matrix::rsparsematrix(n, p, density = density))
  colnames(X) <- paste0("c", seq_len(ncol(X)))
  X
}

# Column-normalised dense matrix, suitable for Jensen-Shannon.
make_P <- function(...) {
  P <- as.matrix(make_X(...))
  keep <- colSums(P) > 0
  P <- P[, keep, drop = FALSE]
  sweep(P, 2, colSums(P), "/")
}

test_that("the default call works", {
  # regression: the old default method ("jaccard") was not among the
  # match.arg choices, so the bare call errored.
  X <- make_X()
  expect_silent(d <- sparseDist(X, ncores = 1, verbose = FALSE))
  expect_equal(dim(d), c(ncol(X), ncol(X)))
})

test_that("'jaccard' is accepted as an alias for 'binary'", {
  X <- make_X()
  expect_equal(
    as.matrix(sparseDist(X, method = "jaccard", ncores = 1, verbose = FALSE)),
    as.matrix(sparseDist(X, method = "binary",  ncores = 1, verbose = FALSE))
  )
})

test_that("similarity results have a unit diagonal for any 'diag'", {
  # regression: the i == j branch was only reachable when diag = TRUE, so
  # similarity results were left with a 0 diagonal (0 self-similarity).
  X <- make_X()
  for (m in c("binary", "pearson")) {
    for (dg in c(TRUE, FALSE)) {
      s <- sparseDist(X, method = m, dist = FALSE, diag = dg,
                      ncores = 1, verbose = FALSE)
      expect_equal(diag(as.matrix(s)), rep(1, ncol(X)),
                   ignore_attr = TRUE,
                   info = paste(m, "diag =", dg))
    }
  }
})

test_that("distance results have an exact zero diagonal", {
  X <- make_X()
  for (m in c("binary", "euclidean", "manhattan", "pearson")) {
    d <- sparseDist(X, method = m, diag = TRUE, full = TRUE,
                    ncores = 1, verbose = FALSE)
    # diag() carries the matrix dimnames; drop them so identical() compares
    # the values only. unname() preserves exactness (unlike a tolerance).
    expect_identical(unname(diag(as.matrix(d))), rep(0, ncol(X)),
                     info = m)
  }
  # js is validated separately: it needs non-negative, normalised input
  P <- make_P()
  d <- sparseDist(P, method = "js", diag = TRUE, full = TRUE,
                  ncores = 1, verbose = FALSE)
  expect_identical(unname(diag(d)), rep(0, ncol(P)))
})

test_that("storage follows the result: dense distances, sparse coefficients", {
  X <- make_X()
  expect_true(is.matrix(sparseDist(X, method = "euclidean", ncores = 1, verbose = FALSE)))
  expect_true(is.matrix(sparseDist(X, method = "manhattan", ncores = 1, verbose = FALSE)))
  expect_true(is.matrix(sparseDist(X, method = "binary", dist = TRUE, ncores = 1, verbose = FALSE)))
  expect_s4_class(sparseDist(X, method = "binary", dist = FALSE, ncores = 1, verbose = FALSE), "dgCMatrix")
  expect_s4_class(sparseDist(X, method = "covariance", ncores = 1, verbose = FALSE), "dgCMatrix")
})

test_that("two-matrix forms run for every method", {
  # regression: pearson/js/covariance passed the wrong density (dense vs
  # sparse) to their kernels and errored whenever Y was supplied.
  X <- make_X(); Y <- X[, 1:3]
  for (m in c("binary", "euclidean", "manhattan", "pearson", "covariance")) {
    r <- sparseDist(X, Y, method = m, ncores = 1, verbose = FALSE)
    expect_equal(dim(r), c(ncol(X), ncol(Y)), info = m)
  }
  P <- make_P(); Q <- P[, 1:3, drop = FALSE]
  expect_equal(dim(sparseDist(P, Q, method = "js", ncores = 1, verbose = FALSE)),
               c(ncol(P), ncol(Q)))
})

test_that("dimension names are propagated", {
  X <- make_X(); Y <- X[, 1:3]
  d <- sparseDist(X, ncores = 1, verbose = FALSE)
  expect_equal(rownames(d), colnames(X))
  expect_equal(colnames(d), colnames(X))
  r <- sparseDist(X, Y, method = "euclidean", ncores = 1, verbose = FALSE)
  expect_equal(rownames(r), colnames(X))
  expect_equal(colnames(r), colnames(Y))
})

test_that("Jensen-Shannon is the standard metric", {
  # sqrt(JSD), bounded above by sqrt(log 2); the pre-fix kernel returned
  # sqrt(2 * JSD), i.e. sqrt(2) times larger.
  P <- cbind(a = c(1, 0), b = c(0, 1))
  d <- sparseDist(P, method = "js", full = TRUE, diag = TRUE,
                  ncores = 1, verbose = FALSE)
  expect_equal(d[1, 2], sqrt(log(2)))
  expect_identical(unname(d[1, 1]), 0)

  # every pair of normalised columns stays within the bound
  Pn <- make_P()
  dn <- sparseDist(Pn, method = "js", full = TRUE, diag = TRUE,
                   ncores = 1, verbose = FALSE)
  expect_true(all(dn <= sqrt(log(2)) + 1e-12))
  expect_true(all(dn >= 0))
})

test_that("Jensen-Shannon rejects invalid input", {
  expect_error(sparseDist(matrix(c(-1, 2, 3, 4), 2), method = "js",
                          ncores = 1, verbose = FALSE),
               "non-negative")
  expect_error(sparseDist(matrix(c(NA_real_, 2, 3, 4), 2), method = "js",
                          ncores = 1, verbose = FALSE),
               "finite")
})

test_that("correlation distance stays within [0, 2]", {
  # regression: arma::cor could return a value a few ULP outside [-1, 1],
  # making 1 - c a tiny NEGATIVE distance.
  X <- make_X()
  d <- sparseDist(X, method = "pearson", full = TRUE, diag = TRUE,
                  ncores = 1, verbose = FALSE)
  expect_true(all(d >= 0))
  expect_true(all(d <= 2))
  # a column against itself, via the two-matrix path
  r <- sparseDist(X, X[, 1:3], method = "pearson", ncores = 1, verbose = FALSE)
  expect_true(all(r >= 0))
})

test_that("'full' controls symmetry consistently across methods", {
  X <- make_X()
  for (m in c("binary", "euclidean", "manhattan", "pearson", "covariance")) {
    full <- as.matrix(sparseDist(X, method = m, full = TRUE, diag = TRUE,
                                 ncores = 1, verbose = FALSE))
    expect_equal(full, t(full), ignore_attr = TRUE, info = m)

    low <- as.matrix(sparseDist(X, method = m, full = FALSE, diag = TRUE,
                                ncores = 1, verbose = FALSE))
    expect_true(all(low[upper.tri(low)] == 0), info = m)
    expect_equal(low[lower.tri(low)], full[lower.tri(full)],
                 ignore_attr = TRUE, info = m)
  }
})

test_that("euclidean and manhattan agree with base R", {
  X <- make_X()
  Xd <- as.matrix(X)
  ours <- sparseDist(X, method = "euclidean", full = TRUE, diag = TRUE,
                     ncores = 1, verbose = FALSE)
  ref  <- as.matrix(dist(t(Xd), method = "euclidean"))
  expect_equal(ours, ref, ignore_attr = TRUE)

  ours <- sparseDist(X, method = "manhattan", full = TRUE, diag = TRUE,
                     ncores = 1, verbose = FALSE)
  ref  <- as.matrix(dist(t(Xd), method = "manhattan"))
  expect_equal(ours, ref, ignore_attr = TRUE)
})

test_that("pearson and covariance agree with base R", {
  X <- make_X()
  Xd <- as.matrix(X)
  ours <- as.matrix(sparseDist(X, method = "pearson", dist = FALSE, full = TRUE,
                               diag = TRUE, ncores = 1, verbose = FALSE))
  expect_equal(ours, cor(Xd), ignore_attr = TRUE)

  ours <- as.matrix(sparseDist(X, method = "covariance", full = TRUE,
                               diag = TRUE, ncores = 1, verbose = FALSE))
  expect_equal(ours, cov(Xd), ignore_attr = TRUE)
})

test_that("jaccard matches a direct set computation", {
  X <- make_X()
  Xd <- as.matrix(X) != 0
  p <- ncol(Xd)
  ref <- matrix(0, p, p)
  for (i in seq_len(p)) for (j in seq_len(p)) {
    inter <- sum(Xd[, i] & Xd[, j])
    uni   <- sum(Xd[, i] | Xd[, j])
    ref[i, j] <- if (uni > 0) inter / uni else 1
  }
  ours <- as.matrix(sparseDist(X, method = "binary", dist = FALSE, full = TRUE,
                               diag = TRUE, ncores = 1, verbose = FALSE))
  expect_equal(ours, ref, ignore_attr = TRUE)
})

test_that("dense, data.frame and sparse inputs agree", {
  X <- make_X()
  ref <- sparseDist(X, method = "euclidean", ncores = 1, verbose = FALSE)
  expect_equal(sparseDist(as.matrix(X), method = "euclidean",
                          ncores = 1, verbose = FALSE), ref)
  expect_equal(sparseDist(as.data.frame(as.matrix(X)), method = "euclidean",
                          ncores = 1, verbose = FALSE), ref, ignore_attr = TRUE)
})

test_that("mismatched row counts are rejected", {
  X <- make_X(n = 60)
  Z <- make_X(n = 50, seed = 2)
  expect_error(sparseDist(X, Z, method = "euclidean", ncores = 1, verbose = FALSE),
               "Mismatched")
})

test_that("degenerate columns do not produce NaN", {
  X <- as.matrix(make_X())
  X[, 1] <- 0                     # empty column: undefined correlation
  X[, 2] <- 1                     # constant column: zero variance
  Xs <- methods::as(X, "CsparseMatrix")
  for (m in c("binary", "euclidean", "manhattan", "pearson", "covariance")) {
    d <- sparseDist(Xs, method = m, full = TRUE, diag = TRUE,
                    ncores = 1, verbose = FALSE)
    expect_false(anyNA(as.matrix(d)), info = m)
  }
})

test_that("ncores is sanitised", {
  X <- make_X()
  ref <- sparseDist(X, method = "euclidean", ncores = 1, verbose = FALSE)
  expect_equal(sparseDist(X, method = "euclidean", ncores = 0, verbose = FALSE), ref)
  expect_equal(sparseDist(X, method = "euclidean", ncores = -5, verbose = FALSE), ref)
  expect_equal(sparseDist(X, method = "euclidean", ncores = 2, verbose = FALSE), ref)
})
