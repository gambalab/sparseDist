# Tests for the shared-neighbour edge set (edges = "shared").
#
# These use a HAND-BUILT index matrix rather than a sparseKNN() result, because
# sparseKNN() can only ever produce regular fixed-size neighbourhoods. The
# interesting behaviour lives in the irregular cases it cannot generate:
# NA padding, duplicate neighbours, an explicit self index, and neighbour sets
# of differing cardinality -- the last being precisely where the true-union
# formula diverges from Seurat's 2k - overlap.
#
# All calls pass ncores = 1.

# A deliberately awkward neighbourhood system:
#
#   row 1: {4, 2}      NA padding
#   row 2: {4, 1}      NA padding
#   row 3: {4, 5, 5}   duplicate neighbour
#   row 4: {4, 5, 6}   EXPLICIT SELF INDEX
#   row 5: {6}         NA padding, short row
#   row 6: {5}         NA padding, short row
#
# With include_self = TRUE the sets are
#   N(1) = N(2) = {1,2,4}; N(3) = {3,4,5}; N(4) = {4,5,6}; N(5) = N(6) = {5,6}
# so pairs (1,3), (2,3), (3,6) share a neighbour WITHOUT being kNN-adjacent --
# the case that distinguishes the two edge sets -- and node 4 carries a
# self-loop that "shared" can never emit.
awkward_idx <- function() {
  idx <- rbind(
    c(4L, 2L, NA),
    c(4L, 1L, NA),
    c(4L, 5L, 5L),
    c(4L, 5L, 6L),
    c(6L, NA, NA),
    c(5L, NA, NA)
  )
  rownames(idx) <- paste0("n", seq_len(nrow(idx)))
  idx
}

# Neighbour sets by definition; include_self decides membership outright.
ref_sets <- function(idx, include_self = TRUE) {
  lapply(seq_len(nrow(idx)), function(i) {
    s <- idx[i, ]
    s <- s[!is.na(s)]
    s <- s[s != i]
    if (include_self) s <- c(i, s)
    sort(unique(s))
  })
}

# Every pair sharing at least one neighbour, weighted by the TRUE union.
# Deliberately O(n^2) and written straight from the definition: it is the
# independent check on the inverted-index traversal, so it must not share any
# of its machinery.
ref_shared <- function(idx, include_self = TRUE, prune = 0) {
  n <- nrow(idx)
  sets <- ref_sets(idx, include_self)
  out <- matrix(0, n, n, dimnames = list(rownames(idx), rownames(idx)))
  for (i in seq_len(n - 1L)) {
    for (j in seq(i + 1L, n)) {
      a <- sets[[i]]; b <- sets[[j]]
      if (!length(a) || !length(b)) next
      inter <- length(intersect(a, b))
      if (inter == 0L) next
      w <- inter / length(union(a, b))
      if (w > prune) { out[i, j] <- w; out[j, i] <- w }
    }
  }
  out
}

test_that("shared weights match a direct O(n^2) set computation", {
  idx <- awkward_idx()
  for (self in c(TRUE, FALSE)) {
    for (pr in c(0, 0.2, 0.5)) {
      got <- as.matrix(sparseSNN(idx, include_self = self, prune = pr,
                                 edges = "shared", ncores = 1,
                                 verbose = FALSE))
      want <- ref_shared(idx, include_self = self, prune = pr)
      expect_equal(got, want,
                   info = paste("include_self =", self, "prune =", pr))
    }
  }
})

test_that("the shared edge set has the expected structure", {
  idx <- awkward_idx()
  g <- sparseSNN(idx, edges = "shared", prune = 0, ncores = 1, verbose = FALSE)
  m <- as.matrix(g)

  # 11 undirected pairs share a neighbour; see the fixture comment.
  expect_equal(sum(m != 0) / 2, 11)
  expect_true(isSymmetric(m))
  # "shared" emits only j > i, so a diagonal entry is structurally impossible.
  expect_true(all(diag(m) == 0))
  # N(5) == N(6) == {5,6}, so their overlap is total.
  expect_equal(unname(m["n5", "n6"]), 1)
  # N(1) and N(3) share only node 4, out of a five-element union.
  expect_equal(unname(m["n1", "n3"]), 0.2)
})

test_that("knn off-diagonal edges are contained in shared, self-loops are not", {
  idx <- awkward_idx()
  knn <- as.matrix(sparseSNN(idx, edges = "knn", prune = 0, ncores = 1,
                             verbose = FALSE))
  shr <- as.matrix(sparseSNN(idx, edges = "shared", prune = 0, ncores = 1,
                             verbose = FALSE))

  off <- function(m) { diag(m) <- 0; m }
  # Containment holds for the OFF-DIAGONAL edges only.
  expect_true(all(off(shr)[off(knn) != 0] != 0))
  # And here it is proper: (1,3), (2,3) and (3,6) share a neighbour without
  # being kNN-adjacent.
  expect_gt(sum(off(shr) != 0), sum(off(knn) != 0))
  expect_equal(sum(off(knn) != 0) / 2, 8)

  # Row 4 lists its own index, so the kNN edge set carries a self-loop of
  # weight 1. The shared edge set never does.
  expect_equal(unname(knn["n4", "n4"]), 1)
  expect_true(all(diag(shr) == 0))

  # Wherever both emit an edge the weights agree exactly.
  both <- off(knn) != 0 & off(shr) != 0
  expect_equal(off(knn)[both], off(shr)[both])
})

test_that("in-kernel pruning matches pruning after the fact", {
  idx <- awkward_idx()
  full <- as.matrix(sparseSNN(idx, edges = "shared", prune = 0, ncores = 1,
                              verbose = FALSE))
  for (pr in c(0, 0.2, 0.25, 0.5, 2 / 3)) {
    # The kernel drops edges before they are ever materialised; doing it here
    # instead must give the identical matrix, including the strict test.
    want <- full
    want[want <= pr] <- 0
    got <- as.matrix(sparseSNN(idx, edges = "shared", prune = pr, ncores = 1,
                               verbose = FALSE))
    expect_equal(got, want, info = paste("prune =", pr))
  }
})

test_that("prune = 1 yields an empty graph without error", {
  idx <- awkward_idx()
  # Exercises the early-return path: pass one finds nothing, so pass two is
  # skipped entirely.
  g <- sparseSNN(idx, edges = "shared", prune = 1, ncores = 1, verbose = FALSE)
  expect_s4_class(g, "dgCMatrix")
  expect_equal(dim(g), c(6L, 6L))
  expect_equal(Matrix::nnzero(g), 0L)
})

test_that("include_self changes the sets and therefore the weights", {
  idx <- awkward_idx()
  with_self <- as.matrix(sparseSNN(idx, include_self = TRUE, edges = "shared",
                                   ncores = 1, verbose = FALSE))
  no_self <- as.matrix(sparseSNN(idx, include_self = FALSE, edges = "shared",
                                 ncores = 1, verbose = FALSE))
  expect_false(identical(with_self, no_self))
  # Membership is decided outright, so row 4's explicit self index is removed
  # when include_self = FALSE rather than left in place.
  expect_equal(no_self, ref_shared(idx, include_self = FALSE, prune = 0))
})

test_that("symmetrise is ignored for the shared edge set", {
  idx <- awkward_idx()
  expect_warning(
    g <- sparseSNN(idx, edges = "shared", symmetrise = FALSE, ncores = 1,
                   verbose = FALSE),
    "symmetrise"
  )
  expect_true(isSymmetric(as.matrix(g)))
  # No warning when it is simply left at its default.
  expect_silent(sparseSNN(idx, edges = "shared", ncores = 1, verbose = FALSE))
})

test_that("ragged neighbourhoods use the true union, not 2k - overlap", {
  idx <- awkward_idx()
  g <- as.matrix(sparseSNN(idx, edges = "shared", ncores = 1, verbose = FALSE))

  # N(3) = {3,4,5} and N(5) = {5,6}: overlap 1, true union 4, so w = 0.25.
  # Seurat's 2k - overlap would use a fixed k and give a different denominator
  # here, which is exactly why this fixture has rows of differing length.
  expect_equal(unname(g["n3", "n5"]), 0.25)

  # N(4) = {4,5,6} and N(5) = {5,6}: overlap 2, true union 3.
  expect_equal(unname(g["n4", "n5"]), 2 / 3)
})

test_that("shared mode rejects the same malformed input as knn mode", {
  idx <- awkward_idx()
  bad <- idx; bad[1, 1] <- 99L
  expect_error(sparseSNN(bad, edges = "shared", ncores = 1, verbose = FALSE),
               "1-based row indices")
  frac <- idx; storage.mode(frac) <- "double"; frac[1, 1] <- 1.5
  expect_error(sparseSNN(frac, edges = "shared", ncores = 1, verbose = FALSE),
               "whole-number")
})

test_that("shared mode agrees with the reference on a sparseKNN result", {
  # Regular neighbourhoods, as a cross-check that the awkward fixture is not
  # the only thing being exercised.
  set.seed(1)
  X <- abs(Matrix::rsparsematrix(60, 12, density = 0.3))
  colnames(X) <- paste0("c", seq_len(ncol(X)))
  nn <- sparseKNN(X, k = 4, method = "binary", include_self = FALSE,
                  ncores = 1, verbose = FALSE)

  got <- as.matrix(sparseSNN(nn, edges = "shared", prune = 0, ncores = 1,
                             verbose = FALSE))
  want <- ref_shared(nn$idx, include_self = TRUE, prune = 0)
  dimnames(want) <- dimnames(got)
  expect_equal(got, want)
})
