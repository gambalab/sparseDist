# Tests for the blocked k-nearest-neighbour search.
#
# The strongest property is that blocking must not change the answer: every
# result is cross-checked against the full matrix from sparseDist(), which is
# itself already tested. All calls pass ncores = 1 for CRAN's core limit.

make_X <- function(n = 60, p = 12, density = 0.3, seed = 1) {
  set.seed(seed)
  X <- abs(Matrix::rsparsematrix(n, p, density = density))
  colnames(X) <- paste0("c", seq_len(ncol(X)))
  X
}

# Reference top-k taken from the full dense matrix, with the same tie rule
# (lower index wins) that the C++ heap uses.
ref_topk <- function(M, k, decreasing, exclude_self) {
  nref <- nrow(M)          # candidates live in the ROWS of M
  nq   <- ncol(M)          # one result row per QUERY column
  # self-exclusion only makes sense when M is a square self-comparison
  if (exclude_self) stopifnot(nref == nq)
  idx <- matrix(NA_integer_, nq, k)
  val <- matrix(NA_real_, nq, k)
  for (q in seq_len(nq)) {
    v <- M[, q]
    cand <- seq_len(nref)
    if (exclude_self) cand <- cand[cand != q]
    cand <- cand[is.finite(v[cand])]
    ord <- cand[order(if (decreasing) -v[cand] else v[cand], cand)]
    take <- head(ord, k)
    if (length(take)) {
      idx[q, seq_along(take)] <- take
      val[q, seq_along(take)] <- v[take]
    }
  }
  list(idx = idx, dist = val)
}

test_that("blocked search matches the full matrix", {
  X <- make_X()
  for (m in c("binary", "euclidean", "manhattan", "pearson")) {
    full <- as.matrix(sparseDist(X, method = m, full = TRUE, diag = TRUE,
                                 ncores = 1, verbose = FALSE))
    ref  <- ref_topk(full, k = 4, decreasing = FALSE, exclude_self = TRUE)
    got  <- sparseKNN(X, k = 4, method = m, ncores = 1, verbose = FALSE)
    expect_equal(unname(got$idx),  ref$idx,  info = m)
    expect_equal(unname(got$dist), ref$dist, info = m)
  }
})

test_that("the block size does not change the result", {
  X <- make_X(p = 20)
  base <- sparseKNN(X, k = 5, block_size = 20L, ncores = 1, verbose = FALSE)
  for (bs in c(1L, 2L, 3L, 7L, 100L)) {
    got <- sparseKNN(X, k = 5, block_size = bs, ncores = 1, verbose = FALSE)
    expect_equal(got, base, info = paste("block_size =", bs))
  }
})

test_that("similarity mode keeps the largest values", {
  X <- make_X()
  full <- as.matrix(sparseDist(X, method = "binary", dist = FALSE,
                               full = TRUE, diag = TRUE,
                               ncores = 1, verbose = FALSE))
  ref <- ref_topk(full, k = 4, decreasing = TRUE, exclude_self = TRUE)
  got <- sparseKNN(X, k = 4, method = "binary", dist = FALSE,
                   ncores = 1, verbose = FALSE)
  expect_equal(unname(got$idx),  ref$idx)
  expect_equal(unname(got$dist), ref$dist)
  # values must be non-increasing along each row
  expect_true(all(apply(got$dist, 1, function(r) all(diff(r) <= 0))))
})

test_that("distance mode returns non-decreasing rows", {
  X <- make_X()
  got <- sparseKNN(X, k = 5, method = "euclidean", ncores = 1, verbose = FALSE)
  expect_true(all(apply(got$dist, 1, function(r) all(diff(r) >= 0))))
})

test_that("include_self controls self-neighbours", {
  X <- make_X()
  off <- sparseKNN(X, k = 3, method = "euclidean", include_self = FALSE,
                   ncores = 1, verbose = FALSE)
  # no column may appear as its own neighbour
  expect_false(any(off$idx == row(off$idx), na.rm = TRUE))

  on <- sparseKNN(X, k = 3, method = "euclidean", include_self = TRUE,
                  ncores = 1, verbose = FALSE)
  # a column is at distance 0 from itself, so it must be its own first neighbour
  expect_equal(on$idx[, 1], seq_len(ncol(X)), ignore_attr = TRUE)
  expect_equal(on$dist[, 1], rep(0, ncol(X)), ignore_attr = TRUE)
})

test_that("a reference matrix is searched instead of X", {
  X <- make_X(p = 12)
  Y <- X[, 1:5]
  got <- sparseKNN(X, Y, k = 3, method = "euclidean", ncores = 1, verbose = FALSE)
  expect_equal(dim(got$idx), c(ncol(X), 3L))
  expect_true(all(got$idx >= 1 & got$idx <= ncol(Y)))

  full <- as.matrix(sparseDist(Y, X, method = "euclidean",
                               ncores = 1, verbose = FALSE))
  ref <- ref_topk(full, k = 3, decreasing = FALSE, exclude_self = FALSE)
  expect_equal(unname(got$idx), ref$idx)
})

test_that("dimension names and shape are as documented", {
  X <- make_X()
  got <- sparseKNN(X, k = 4, ncores = 1, verbose = FALSE)
  expect_equal(rownames(got$idx),  colnames(X))
  expect_equal(rownames(got$dist), colnames(X))
  expect_equal(dim(got$idx), c(ncol(X), 4L))
  expect_type(got$idx, "integer")
  expect_type(got$dist, "double")
})

test_that("k larger than the candidate pool is clamped with a warning", {
  X <- make_X(p = 5)
  expect_warning(got <- sparseKNN(X, k = 99, ncores = 1, verbose = FALSE),
                 "exceeds")
  expect_equal(ncol(got$idx), 4L)   # 5 columns, self excluded
})

test_that("invalid arguments are rejected", {
  X <- make_X()
  expect_error(sparseKNN(X, k = 0, ncores = 1, verbose = FALSE), "positive")
  expect_error(sparseKNN(X, k = -1, ncores = 1, verbose = FALSE), "positive")
  expect_error(sparseKNN(X, block_size = 0L, ncores = 1, verbose = FALSE), "positive")
})

test_that("count arguments must be single finite whole numbers", {
  X <- make_X()
  # fractional values used to be truncated silently
  expect_error(sparseKNN(X, k = 2.5, ncores = 1, verbose = FALSE), "positive integer")
  expect_error(sparseKNN(X, block_size = 4.5, ncores = 1, verbose = FALSE), "positive integer")
  expect_error(sparseKNN(X, k = 3, ncores = 1.7, verbose = FALSE), "non-negative integer")
  # Inf became NA_integer_ and failed with an unrelated error
  expect_error(sparseKNN(X, k = 3, ncores = Inf, verbose = FALSE), "non-negative integer")
  expect_error(sparseKNN(X, k = Inf, ncores = 1, verbose = FALSE), "positive integer")
  # vectors and empty values
  expect_error(sparseKNN(X, k = c(2, 3), ncores = 1, verbose = FALSE), "positive integer")
  expect_error(sparseKNN(X, k = integer(0), ncores = 1, verbose = FALSE), "positive integer")
  expect_error(sparseKNN(X, k = "3", ncores = 1, verbose = FALSE), "positive integer")
})

test_that("results are independent of the thread count", {
  X <- make_X(p = 20)
  a <- sparseKNN(X, k = 5, ncores = 1, verbose = FALSE)
  b <- sparseKNN(X, k = 5, ncores = 2, verbose = FALSE)
  expect_identical(a, b)
})

test_that("dense and sparse inputs agree", {
  X <- make_X()
  a <- sparseKNN(X, k = 4, method = "euclidean", ncores = 1, verbose = FALSE)
  b <- sparseKNN(as.matrix(X), k = 4, method = "euclidean",
                 ncores = 1, verbose = FALSE)
  expect_equal(a, b)
})

test_that("ties are broken by the lower column index", {
  # identical columns: every pairwise distance is 0, so the choice among them
  # is decided entirely by the tie rule.
  X <- Matrix::Matrix(matrix(rep(c(1, 0, 1, 0), 6), nrow = 4), sparse = TRUE)
  colnames(X) <- paste0("c", seq_len(ncol(X)))
  got <- sparseKNN(X, k = 3, method = "euclidean", ncores = 1, verbose = FALSE)
  # for column 4, the nearest are columns 1, 2, 3 in that order
  expect_equal(unname(got$idx[4, ]), c(1L, 2L, 3L))
})

test_that("the result rebuilds a sparse adjacency matrix", {
  X <- make_X()
  got <- sparseKNN(X, k = 3, method = "binary", dist = FALSE,
                   ncores = 1, verbose = FALSE)
  A <- Matrix::sparseMatrix(i = as.vector(row(got$idx)),
                            j = as.vector(got$idx),
                            x = as.vector(got$dist),
                            dims = c(ncol(X), ncol(X)))
  expect_equal(dim(A), c(ncol(X), ncol(X)))
  expect_equal(length(A@x), ncol(X) * 3L)
})

test_that("missing neighbours are padded with NA, not NaN", {
  # only one finite candidate, so slots 2 and 3 cannot be filled
  D <- matrix(c(1, NA_real_, Inf), ncol = 1)
  got <- sparseDist:::topKBlock(D = D, k = 3, decreasing = FALSE,
                                self_row = 0L, ncores = 1)
  expect_true(is.na(got$dist[1, 2]))
  expect_false(is.nan(got$dist[1, 2]))
  expect_identical(got$dist[1, 2], NA_real_)
  expect_true(is.na(got$idx[1, 2]))
  # the one valid candidate is still returned
  expect_equal(got$idx[1, 1], 1L)
  expect_equal(got$dist[1, 1], 1)
})

test_that("topKBlock validates self_row", {
  D <- matrix(runif(6), nrow = 3)
  expect_error(sparseDist:::topKBlock(D, 2, FALSE, c(1L, 2L, 3L), 1),
               "one entry per column")
  expect_error(sparseDist:::topKBlock(D, 2, FALSE, c(1L, 99L), 1),
               "valid 1-based rows")
  expect_error(sparseDist:::topKBlock(D, 2, FALSE, c(1L, -1L), 1),
               "valid 1-based rows")
  expect_error(sparseDist:::topKBlock(D, 0, FALSE, c(0L, 0L), 1),
               "k must be")
})

test_that("ranking direction follows the method, not just 'dist'", {
  X <- make_X()

  # distance-only methods refuse dist = FALSE rather than silently returning
  # the FARTHEST columns
  for (m in c("euclidean", "manhattan", "js")) {
    expect_error(sparseKNN(X, k = 3, method = m, dist = FALSE,
                           ncores = 1, verbose = FALSE),
                 "only meaningful", info = m)
  }

  # covariance is a coefficient: the largest values are the neighbours, even
  # though dist defaults to TRUE
  full <- as.matrix(sparseDist(X, method = "covariance", full = TRUE,
                               diag = TRUE, ncores = 1, verbose = FALSE))
  ref <- ref_topk(full, k = 3, decreasing = TRUE, exclude_self = TRUE)
  got <- sparseKNN(X, k = 3, method = "covariance", ncores = 1, verbose = FALSE)
  expect_equal(unname(got$idx), ref$idx)
  expect_true(all(apply(got$dist, 1, function(r) all(diff(r) <= 0))))
})

test_that("mismatched row counts are rejected", {
  X <- make_X(n = 60, p = 8)
  Z <- make_X(n = 50, p = 8, seed = 3)
  expect_error(sparseKNN(X, Z, k = 3, ncores = 1, verbose = FALSE),
               "same number of rows")
})

test_that("'jaccard' is accepted as an alias", {
  X <- make_X()
  expect_equal(sparseKNN(X, k = 3, method = "jaccard", ncores = 1, verbose = FALSE),
               sparseKNN(X, k = 3, method = "binary",  ncores = 1, verbose = FALSE))
})

test_that("logical flags are validated as scalars", {
  X <- make_X()
  for (bad in list(NA, 1, "TRUE", c(TRUE, FALSE), logical(0))) {
    expect_error(sparseKNN(X, dist = bad, ncores = 1, verbose = FALSE),
                 "TRUE or FALSE")
    expect_error(sparseKNN(X, include_self = bad, ncores = 1, verbose = FALSE),
                 "TRUE or FALSE")
  }
  expect_error(sparseKNN(X, ncores = 1, verbose = NA), "TRUE or FALSE")
  expect_error(sparseKNN(X, ncores = NA, verbose = FALSE), "non-negative integer")
})

test_that("self-search keeps exact self values for every method", {
  # a constant column makes correlation undefined, which the rectangular
  # kernel maps to 0; the self entry must still be restored to 1 / 0.
  X <- Matrix::Matrix(cbind(constant = rep(1, 5), varying = 1:5), sparse = TRUE)

  sim <- sparseKNN(X, k = 2, method = "pearson", dist = FALSE,
                   include_self = TRUE, ncores = 1, verbose = FALSE)
  self_at <- match(1L, sim$idx[1, ])
  expect_false(is.na(self_at))                     # the self column is present
  # unname(): the extracted value carries the row name of the matrix
  expect_equal(unname(sim$dist[1, self_at]), 1)

  dst <- sparseKNN(X, k = 2, method = "pearson", dist = TRUE,
                   include_self = TRUE, ncores = 1, verbose = FALSE)
  expect_equal(dst$idx[, 1], 1:2, ignore_attr = TRUE)   # self is nearest
  expect_equal(dst$dist[, 1], c(0, 0), ignore_attr = TRUE)

  # JS self-distance must be an exact 0, not sqrt(round-off) ~ 3e-9
  P <- as.matrix(abs(make_X(p = 6)))
  P <- sweep(P, 2, colSums(P), "/")
  js <- sparseKNN(P, k = 2, method = "js", include_self = TRUE,
                  ncores = 1, verbose = FALSE)
  expect_identical(unname(js$dist[, 1]), rep(0, ncol(P)))
  expect_equal(js$idx[, 1], seq_len(ncol(P)), ignore_attr = TRUE)

  # covariance is the documented exception: the self value is the variance.
  # Ask for every candidate (k = ncol) so the self column is guaranteed to be
  # present -- with a smaller k it may not rank, leaving nothing to compare
  # and letting the assertion pass vacuously.
  X2 <- make_X(p = 6)
  cv <- sparseKNN(X2, k = ncol(X2), method = "covariance", include_self = TRUE,
                  ncores = 1, verbose = FALSE)
  vars <- apply(as.matrix(X2), 2, var)
  self_at <- vapply(seq_len(ncol(X2)),
                    function(i) match(i, cv$idx[i, ]), integer(1))
  expect_false(anyNA(self_at))
  expect_equal(cv$dist[cbind(seq_len(ncol(X2)), self_at)],
               unname(vars), ignore_attr = TRUE)
})

test_that("ncores = 0 is resolved and does not change results", {
  # .detect_cores() would otherwise spin up every core minus one during
  # R CMD check, and CRAN allows at most two.
  testthat::local_mocked_bindings(.detect_cores = function() 1L,
                                  .package = "sparseDist")
  X <- make_X(p = 12)
  expect_equal(sparseKNN(X, k = 3, ncores = 0, verbose = FALSE),
               sparseKNN(X, k = 3, ncores = 1, verbose = FALSE))
})

test_that("verbose draws a progress bar and FALSE is silent", {
  X <- make_X(p = 12)
  expect_silent(sparseKNN(X, k = 3, ncores = 1, verbose = FALSE))
  # the bar goes to the console, not to message(); capture it as output
  out <- capture.output(
    sparseKNN(X, k = 3, block_size = 4L, ncores = 1, verbose = TRUE)
  )
  expect_true(any(grepl("=|%", out)))
})

test_that("counts above .Machine$integer.max are rejected", {
  X <- make_X()
  # 3e9 is a finite whole number, so it survives every other check, but
  # as.integer() would silently turn it into NA_integer_
  expect_error(sparseKNN(X, k = 3e9, ncores = 1, verbose = FALSE),
               "integer.max")
  expect_error(sparseKNN(X, k = 3, block_size = 3e9, ncores = 1, verbose = FALSE),
               "integer.max")
  expect_error(sparseKNN(X, k = 3, ncores = 3e9, verbose = FALSE),
               "integer.max")
})

test_that("cosine works in the blocked search", {
  X <- make_X()
  full <- as.matrix(sparseDist(X, method = "cosine", full = TRUE, diag = TRUE,
                               ncores = 1, verbose = FALSE))
  ref <- ref_topk(full, k = 4, decreasing = FALSE, exclude_self = TRUE)
  got <- sparseKNN(X, k = 4, method = "cosine", ncores = 1, verbose = FALSE)
  expect_equal(unname(got$idx), ref$idx)
  expect_equal(unname(got$dist), ref$dist)

  # similarity mode keeps the largest
  fsim <- as.matrix(sparseDist(X, method = "cosine", dist = FALSE, full = TRUE,
                               diag = TRUE, ncores = 1, verbose = FALSE))
  rsim <- ref_topk(fsim, k = 4, decreasing = TRUE, exclude_self = TRUE)
  gsim <- sparseKNN(X, k = 4, method = "cosine", dist = FALSE,
                    ncores = 1, verbose = FALSE)
  expect_equal(unname(gsim$idx), rsim$idx)

  # self values are restored exactly in the blocked path
  s <- sparseKNN(X, k = 3, method = "cosine", include_self = TRUE,
                 ncores = 1, verbose = FALSE)
  expect_identical(unname(s$dist[, 1]), rep(0, ncol(X)))
})

test_that("degenerate columns are dropped from the search", {
  X <- as.matrix(make_X(p = 10))
  X[, 2] <- 0                    # two all-zero columns: with the
  X[, 5] <- 0                    # J(empty, empty) = 1 convention they would
  Xs <- methods::as(X, "CsparseMatrix")   # otherwise be perfect neighbours

  nn <- sparseKNN(Xs, k = 3, method = "binary", ncores = 1, verbose = FALSE)

  # they get no neighbours of their own
  expect_true(all(is.na(nn$idx[c(2, 5), ])))
  expect_true(all(is.na(nn$dist[c(2, 5), ])))

  # and are never returned as anyone else's
  expect_false(any(nn$idx[-c(2, 5), ] %in% c(2L, 5L)))

  # every other column still has a full set
  expect_false(anyNA(nn$idx[-c(2, 5), ]))

  # same for cosine, where the convention originates
  cs <- sparseKNN(Xs, k = 3, method = "cosine", ncores = 1, verbose = FALSE)
  expect_true(all(is.na(cs$idx[c(2, 5), ])))
  expect_false(any(cs$idx[-c(2, 5), ] %in% c(2L, 5L)))
})

test_that("drop_degenerate = FALSE keeps zero columns in play", {
  X <- as.matrix(make_X(p = 10))
  X[, 2] <- 0
  Xs <- methods::as(X, "CsparseMatrix")

  # the origin is a legitimate point for a true distance
  nn <- sparseKNN(Xs, k = 3, method = "euclidean", drop_degenerate = FALSE,
                  ncores = 1, verbose = FALSE)
  expect_false(anyNA(nn$idx))
  expect_true(any(nn$idx == 2L))

  # and it is excluded again when the flag is on
  dropped <- sparseKNN(Xs, k = 3, method = "euclidean", drop_degenerate = TRUE,
                       ncores = 1, verbose = FALSE)
  expect_true(all(is.na(dropped$idx[2, ])))
  expect_false(any(dropped$idx[-2, ] == 2L, na.rm = TRUE))
})

test_that("dropped columns become isolated nodes in the SNN graph", {
  X <- as.matrix(make_X(p = 10))
  X[, 3] <- 0
  Xs <- methods::as(X, "CsparseMatrix")

  nn <- sparseKNN(Xs, k = 3, method = "binary", ncores = 1, verbose = FALSE)
  g  <- sparseSNN(nn, ncores = 1, verbose = FALSE)

  expect_equal(sum(g[3, ]), 0)     # no outgoing edges
  expect_equal(sum(g[, 3]), 0)     # and none incoming
  expect_false(anyNA(g@x))
})

test_that("k is clamped against the non-degenerate count", {
  X <- as.matrix(make_X(p = 6))
  X[, 1] <- 0
  X[, 2] <- 0
  Xs <- methods::as(X, "CsparseMatrix")
  # 6 columns, 2 empty, self excluded -> 3 usable neighbours
  expect_warning(nn <- sparseKNN(Xs, k = 5, ncores = 1, verbose = FALSE),
                 "exceeds")
  expect_equal(ncol(nn$idx), 3L)
})

test_that("a signed column summing to zero is not treated as empty", {
  # emptiness is decided on colSums(abs(.)), so cancellation must not count:
  # c(1, -1) has data even though its plain column sum is 0.
  X <- Matrix::Matrix(cbind(cancel = c(1, -1, 0, 0),
                            empty  = c(0,  0, 0, 0),
                            a      = c(1,  0, 1, 0),
                            b      = c(0,  1, 0, 1)), sparse = TRUE)
  expect_equal(sum(X[, "cancel"]), 0)          # sums to zero...
  expect_gt(sum(abs(X[, "cancel"])), 0)        # ...but is not empty

  for (m in c("euclidean", "cosine")) {
    nn <- sparseKNN(X, k = 2, method = m, ncores = 1, verbose = FALSE)
    expect_false(anyNA(nn$idx[1, ]), info = m)         # cancel column kept
    expect_true(all(is.na(nn$idx[2, ])), info = m)     # empty column dropped
    expect_false(any(nn$idx[-2, ] == 2L, na.rm = TRUE), info = m)
    # and the cancelling column is still available as a neighbour
    expect_true(any(nn$idx[-1, ] == 1L, na.rm = TRUE), info = m)
  }
})

test_that("cross search handles degenerate columns on either side", {
  clean <- Matrix::Matrix(cbind(p = c(1, 0, 1, 0), q = c(0, 1, 0, 1),
                                r = c(1, 1, 0, 0)), sparse = TRUE)
  holed <- Matrix::Matrix(cbind(s = c(1, 0, 0, 1), z = c(0, 0, 0, 0),
                                t = c(0, 1, 1, 0)), sparse = TRUE)

  # (a) degenerate in the QUERY matrix only: those rows are all NA, and every
  #     reference column stays available
  a <- sparseKNN(holed, clean, k = 2, ncores = 1, verbose = FALSE)
  expect_equal(dim(a$idx), c(3L, 2L))
  expect_true(all(is.na(a$idx[2, ])))
  expect_false(anyNA(a$idx[-2, ]))
  expect_true(all(a$idx[-2, ] %in% 1:3))

  # (b) degenerate in the REFERENCE matrix only: it is never returned, and no
  #     query row is dropped
  b <- sparseKNN(clean, holed, k = 2, ncores = 1, verbose = FALSE)
  expect_false(anyNA(b$idx))
  expect_false(any(b$idx == 2L))

  # (c) degenerate on both sides: the query row is NA and the reference column
  #     is never used
  cc <- sparseKNN(holed, holed, k = 2, ncores = 1, verbose = FALSE)
  expect_true(all(is.na(cc$idx[2, ])))
  expect_false(any(cc$idx == 2L, na.rm = TRUE))

  # (d) k is capped against the usable REFERENCE columns, not the query count
  expect_warning(d <- sparseKNN(clean, holed, k = 3, ncores = 1,
                                verbose = FALSE), "exceeds")
  expect_equal(ncol(d$idx), 2L)     # 3 reference columns, 1 empty
})
