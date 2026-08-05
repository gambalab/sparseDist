# Tests for the shared-nearest-neighbour graph.
#
# Weights are cross-checked against a direct set computation in R, which is the
# definition the C++ merge walk implements. All calls pass ncores = 1.

make_X <- function(n = 60, p = 12, density = 0.3, seed = 1) {
  set.seed(seed)
  X <- abs(Matrix::rsparsematrix(n, p, density = density))
  colnames(X) <- paste0("c", seq_len(ncol(X)))
  X
}

# Reference: neighbour sets as plain R sets, Jaccard by definition.
ref_snn <- function(idx, include_self = TRUE) {
  n <- nrow(idx); k <- ncol(idx)
  sets <- lapply(seq_len(n), function(i) {
    s <- idx[i, ]
    s <- s[!is.na(s)]
    s <- s[s != i]                 # include_self decides membership outright
    if (include_self) s <- c(i, s)
    sort(unique(s))
  })
  w <- matrix(NA_real_, n, k)
  for (i in seq_len(n)) {
    if (!length(sets[[i]])) next
    for (t in seq_len(k)) {
      v <- idx[i, t]
      if (is.na(v) || !length(sets[[v]])) next
      inter <- length(intersect(sets[[i]], sets[[v]]))
      uni   <- length(union(sets[[i]], sets[[v]]))
      w[i, t] <- inter / uni
    }
  }
  w
}

test_that("weights match a direct set computation", {
  X <- make_X()
  nn <- sparseKNN(X, k = 4, method = "binary", ncores = 1, verbose = FALSE)
  for (self in c(TRUE, FALSE)) {
    got <- sparseDist:::snnJaccard(nn$idx, include_self = self, ncores = 1, verbose = FALSE)
    expect_equal(got, ref_snn(nn$idx, include_self = self),
                 info = paste("include_self =", self))
  }
})

test_that("weights lie in [0, 1] and identical neighbourhoods give 1", {
  X <- make_X()
  nn <- sparseKNN(X, k = 4, ncores = 1, verbose = FALSE)
  w <- sparseDist:::snnJaccard(nn$idx, include_self = TRUE, ncores = 1, verbose = FALSE)
  expect_true(all(w >= 0 & w <= 1, na.rm = TRUE))

  # a point's neighbourhood is identical to itself, so if i lists itself the
  # weight is exactly 1
  idx <- cbind(1:3, c(1L, 2L, 3L))
  w2 <- sparseDist:::snnJaccard(idx, include_self = TRUE, ncores = 1, verbose = FALSE)
  expect_equal(w2[, 2], c(1, 1, 1))
})

test_that("disjoint neighbourhoods give weight 0 and are pruned", {
  # 1 and 2 point at each other but share nothing else
  idx <- rbind(c(2L, 3L), c(1L, 4L), c(1L, 3L), c(2L, 4L))
  w <- sparseDist:::snnJaccard(idx, include_self = FALSE, ncores = 1, verbose = FALSE)
  expect_equal(w, ref_snn(idx, include_self = FALSE))

  g <- sparseSNN(idx, include_self = FALSE, prune = 0,
                 symmetrise = FALSE, ncores = 1, verbose = FALSE)
  expect_true(all(g@x > 0))     # zero-weight edges dropped by the default prune
})

test_that("NA padding is ignored", {
  idx <- rbind(c(2L, 3L), c(1L, NA_integer_), c(1L, 2L))
  w <- sparseDist:::snnJaccard(idx, include_self = TRUE, ncores = 1, verbose = FALSE)
  expect_true(is.na(w[2, 2]))
  expect_equal(w, ref_snn(idx, include_self = TRUE))
  # the padded slot must not become an edge
  g <- sparseSNN(idx, symmetrise = FALSE, ncores = 1, verbose = FALSE)
  expect_false(anyNA(g@x))
})

test_that("out-of-range indices are rejected", {
  expect_error(sparseDist:::snnJaccard(rbind(c(1L, 99L)), TRUE, 1),
               "1-based row indices")
  expect_error(sparseDist:::snnJaccard(rbind(c(1L, 0L)), TRUE, 1),
               "1-based row indices")
  expect_error(sparseDist:::snnJaccard(rbind(c(1L, -2L)), TRUE, 1),
               "1-based row indices")
})

test_that("sparseSNN accepts a sparseKNN result or a bare matrix", {
  X <- make_X()
  nn <- sparseKNN(X, k = 4, ncores = 1, verbose = FALSE)
  a <- sparseSNN(nn, ncores = 1, verbose = FALSE)
  b <- sparseSNN(nn$idx, ncores = 1, verbose = FALSE)
  expect_equal(a, b)
  expect_s4_class(a, "dgCMatrix")
  expect_equal(dim(a), c(ncol(X), ncol(X)))
})

test_that("dimension names are carried through", {
  X <- make_X()
  nn <- sparseKNN(X, k = 4, ncores = 1, verbose = FALSE)
  g <- sparseSNN(nn, ncores = 1, verbose = FALSE)
  expect_equal(rownames(g), colnames(X))
  expect_equal(colnames(g), colnames(X))
})

test_that("prune removes weak edges monotonically", {
  X <- make_X()
  nn <- sparseKNN(X, k = 5, ncores = 1, verbose = FALSE)
  n0 <- length(sparseSNN(nn, prune = 0,   ncores = 1, verbose = FALSE)@x)
  n1 <- length(sparseSNN(nn, prune = 0.2, ncores = 1, verbose = FALSE)@x)
  n2 <- length(sparseSNN(nn, prune = 0.9, ncores = 1, verbose = FALSE)@x)
  expect_true(n0 >= n1)
  expect_true(n1 >= n2)
})

test_that("symmetrise gives an undirected graph", {
  X <- make_X()
  nn <- sparseKNN(X, k = 4, ncores = 1, verbose = FALSE)
  g <- sparseSNN(nn, symmetrise = TRUE, ncores = 1, verbose = FALSE)
  expect_equal(as.matrix(g), t(as.matrix(g)), ignore_attr = TRUE)

  d <- sparseSNN(nn, symmetrise = FALSE, ncores = 1, verbose = FALSE)
  # the directed graph has at most as many edges as the symmetrised one
  expect_true(length(d@x) <= length(g@x))
  # where both directions exist the weights already agree
  M <- as.matrix(d)
  both <- M > 0 & t(M) > 0
  expect_equal(M[both], t(M)[both])
})

test_that("results are independent of the thread count", {
  X <- make_X(p = 20)
  nn <- sparseKNN(X, k = 5, ncores = 1, verbose = FALSE)
  expect_identical(sparseSNN(nn, ncores = 1, verbose = FALSE), sparseSNN(nn, ncores = 2, verbose = FALSE))
})

test_that("arguments are validated", {
  X <- make_X()
  nn <- sparseKNN(X, k = 4, ncores = 1, verbose = FALSE)
  expect_error(sparseSNN(nn, include_self = NA, ncores = 1, verbose = FALSE), "TRUE or FALSE")
  expect_error(sparseSNN(nn, symmetrise = 1, ncores = 1, verbose = FALSE), "TRUE or FALSE")
  expect_error(sparseSNN(nn, prune = NA, ncores = 1, verbose = FALSE),
               "between 0 and 1")
  expect_error(sparseSNN(nn, ncores = 1.5, verbose = FALSE), "non-negative integer")
  expect_error(sparseSNN(nn, ncores = Inf, verbose = FALSE), "non-negative integer")
  expect_error(sparseSNN(list(a = 1), ncores = 1, verbose = FALSE), "sparseKNN")
  expect_error(sparseSNN("nope", ncores = 1, verbose = FALSE), "matrix")
})

test_that("ncores = 0 is resolved", {
  testthat::local_mocked_bindings(.detect_cores = function() 1L,
                                  .package = "sparseDist")
  X <- make_X()
  nn <- sparseKNN(X, k = 4, ncores = 1, verbose = FALSE)
  expect_equal(sparseSNN(nn, ncores = 0, verbose = FALSE), sparseSNN(nn, ncores = 1, verbose = FALSE))
})

test_that("the end-to-end pipeline runs", {
  # distances -> kNN -> SNN graph, the intended workflow
  X <- make_X(p = 25)
  g <- sparseSNN(sparseKNN(X, k = 5, method = "binary",
                           ncores = 1, verbose = FALSE),
                 prune = 1 / 15, ncores = 1, verbose = FALSE)
  expect_s4_class(g, "dgCMatrix")
  expect_equal(dim(g), c(ncol(X), ncol(X)))
  expect_true(all(g@x > 1 / 15))
})

test_that("duplicate neighbour entries do not sum their weights", {
  # sparseMatrix() adds duplicated (i, j) pairs by default, so the same edge
  # emitted twice would become a weight of 2.
  idx <- rbind(c(2L, 2L), c(1L, 1L))
  g <- sparseSNN(idx, include_self = TRUE, symmetrise = FALSE, ncores = 1, verbose = FALSE)
  expect_equal(as.numeric(g[1, 2]), 1)
  expect_equal(as.numeric(g[2, 1]), 1)
  expect_true(all(g@x <= 1))
})

test_that("a cross-reference sparseKNN result is rejected", {
  X <- make_X(p = 12)
  Y <- X[, 1:6]
  cross <- sparseKNN(X, Y, k = 3, ncores = 1, verbose = FALSE)
  expect_error(sparseSNN(cross, ncores = 1, verbose = FALSE), "self-search")

  # the equivalent self-search is accepted
  self <- sparseKNN(X, k = 3, ncores = 1, verbose = FALSE)
  expect_s4_class(sparseSNN(self, ncores = 1, verbose = FALSE), "dgCMatrix")
})

test_that("numeric index matrices are validated, not truncated", {
  expect_error(sparseSNN(matrix(1.5, 1, 1), ncores = 1, verbose = FALSE), "whole-number")
  expect_error(sparseSNN(matrix(Inf, 1, 1), ncores = 1, verbose = FALSE), "finite")
  expect_error(sparseSNN(matrix("1", 1, 1), ncores = 1, verbose = FALSE), "numeric or integer")
  expect_error(sparseSNN(matrix(0L, 1, 1), ncores = 1, verbose = FALSE), "1-based")
  expect_error(sparseSNN(matrix(2L, 1, 1), ncores = 1, verbose = FALSE), "1-based")
  # a whole number stored as double is fine
  expect_s4_class(sparseSNN(matrix(1, 1, 1), ncores = 1, verbose = FALSE), "dgCMatrix")
})

test_that("prune must be a finite value in [0, 1]", {
  X <- make_X()
  nn <- sparseKNN(X, k = 4, ncores = 1, verbose = FALSE)
  for (bad in list(-0.1, 1.5, Inf, -Inf, NA_real_)) {
    expect_error(sparseSNN(nn, prune = bad, ncores = 1, verbose = FALSE), "between 0 and 1")
  }
  # prune = 1 is coherent: the test is strict, so the graph is empty
  expect_equal(length(sparseSNN(nn, prune = 1, ncores = 1, verbose = FALSE)@x), 0L)
})

test_that("the returned object is exactly dgCMatrix", {
  X <- make_X()
  nn <- sparseKNN(X, k = 4, ncores = 1, verbose = FALSE)
  expect_identical(class(sparseSNN(nn, ncores = 1, verbose = FALSE))[1], "dgCMatrix")
})

test_that("include_self overrides self entries already present in idx", {
  # each row lists ITSELF as its first neighbour
  idx <- rbind(c(1L, 2L), c(2L, 3L), c(3L, 1L))

  with_self <- sparseDist:::snnJaccard(idx, include_self = TRUE, ncores = 1)
  without   <- sparseDist:::snnJaccard(idx, include_self = FALSE, ncores = 1)

  # TRUE : S1 = {1,2}, S2 = {2,3} -> 1/3
  expect_equal(with_self[1, 2], 1 / 3)
  # FALSE: S1 = {2},   S2 = {3}   -> 0
  expect_equal(without[1, 2], 0)

  # The weights must not depend on whether the kNN search returned self.
  # Same neighbourhoods, but the self entry is absent; note the edge 1 -> 2
  # sits in column 2 of idx and column 1 of idx_noself.
  idx_noself <- rbind(c(2L, NA_integer_), c(3L, NA_integer_), c(1L, NA_integer_))
  w_noself <- sparseDist:::snnJaccard(idx_noself, include_self = TRUE, ncores = 1)
  expect_equal(with_self[1, 2], w_noself[1, 1])
  expect_equal(with_self[2, 2], w_noself[2, 1])
  expect_equal(with_self[3, 2], w_noself[3, 1])
})

test_that("a list without the self_search marker is rejected", {
  # subsetting drops non-name attributes, so a cross-reference result can
  # arrive unmarked; accepting it would silently build the wrong graph
  X <- make_X(p = 12)
  Y <- X[, 1:6]
  cross <- sparseKNN(X, Y, k = 3, ncores = 1, verbose = FALSE)
  stripped <- cross[c("idx", "dist")]
  expect_null(attr(stripped, "self_search", exact = TRUE))
  expect_error(sparseSNN(stripped, ncores = 1, verbose = FALSE),
               "self_search")

  # the same is true of a hand-built list
  self <- sparseKNN(X, k = 3, ncores = 1, verbose = FALSE)
  expect_error(sparseSNN(list(idx = self$idx), ncores = 1, verbose = FALSE),
               "self_search")
  # ... while the matrix itself, or the intact result, is accepted
  expect_s4_class(sparseSNN(self$idx, ncores = 1, verbose = FALSE), "dgCMatrix")
  expect_s4_class(sparseSNN(self, ncores = 1, verbose = FALSE), "dgCMatrix")
})
