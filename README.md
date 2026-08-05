# sparseDist

Fast distances, nearest-neighbour search and graphs for large, highly sparse
data.

`sparseDist` computes column-wise distances and similarities among the columns
of a sparse matrix (or between the columns of two matrices), in parallel, via
`RcppArmadillo` and OpenMP. For problems too large to materialise a full
`ncol²` matrix, it also provides a memory-bounded blocked nearest-neighbour
search and a shared-nearest-neighbour graph builder.

## Installation

Requires **R ≥ 4.0**.

```r
# from CRAN (once released)
install.packages("sparseDist")

# development version
# install.packages("remotes")
remotes::install_github("gambalab/sparseDist")
```

## Quick start

```r
library(sparseDist)

set.seed(1)
X <- abs(Matrix::rsparsematrix(200, 40, density = 0.3))
colnames(X) <- paste0("c", seq_len(ncol(X)))

# Full pairwise distance matrix (Jaccard distance on the sparsity pattern)
d <- sparseDist(X, ncores = 1, verbose = FALSE)

# The k nearest neighbours of every column, without building the full matrix
nn <- sparseKNN(X, k = 5, method = "cosine", ncores = 1, verbose = FALSE)
head(nn$idx)

# Shared-nearest-neighbour graph, the second stage of graph-based clustering
snn <- sparseSNN(nn, prune = 1/15, ncores = 1, verbose = FALSE)
snn[1:5, 1:5]
```

## What it computes

| Function | Purpose |
|---|---|
| `sparseDist()` | Column-wise distance / similarity matrices |
| `sparseKNN()` | Blocked k-nearest-neighbour search that never materialises the full column-by-column matrix |
| `sparseSNN()` | Shared-nearest-neighbour graph weights from a kNN index, `\|N(i) ∩ N(j)\| / \|N(i) ∪ N(j)\|` |

Methods, shared by all three: `"binary"` (Jaccard on the sparsity pattern, the
default), `"cosine"`, `"euclidean"`, `"manhattan"`, `"pearson"`, `"js"`
(Jensen-Shannon) and `"covariance"`.

## Feature highlights

### `sparseDist()`

- Distances among the columns of one matrix, or a rectangular cross matrix
  between two matrices.
- `"binary"`, `"cosine"` and `"pearson"` have both forms: `dist = TRUE` returns
  the distance, `dist = FALSE` the similarity. The other methods have only one.
- Storage follows the data. Distance-valued results are dense matrices,
  coefficient-valued results are sparse `dgCMatrix`. With sparse input most
  column pairs share nothing, which makes their *distance* non-zero, so a
  distance matrix is nearly full and a sparse representation would cost more
  than it saves. A stored `0` in the distance form therefore unambiguously
  means "identical columns".
- The result is accumulated in a dense buffer inside the kernels. That is
  required for correctness of the parallel fill, not merely for speed:
  inserting into a compressed-sparse structure from several threads would
  corrupt it.

### `sparseKNN()`

- Walks the columns of `X` in blocks, reduces each slab to its `k` best entries
  per column, and returns `ncol(X) × k` `idx` / `dist` matrices — the layout
  used by **FNN** and **RANN**. Peak memory is proportional to
  `ncol(reference) × block_size` rather than `ncol(X)²`.
- Ties at the `k`-th position are broken by the lower column index, so results
  are reproducible and independent of `ncores`.
- Supports a reference matrix `Y`; neighbours are then drawn from its columns.

### `sparseSNN()`

- Reweights a kNN graph by neighbourhood overlap. Accepts a `sparseKNN()`
  result or a bare index matrix, and returns a sparse adjacency matrix.
- `prune` drops edges at or below a weight threshold; `symmetrise` gives an
  undirected graph.

## Comparing embeddings

Cosine is the usual choice for embedding vectors. It is computed by
unit-normalising each column once and taking a merge-walk dot product, so no
column is ever densified and the result is stable even at extreme magnitudes.

```r
# E: features x items, columns are embedding vectors
nn <- sparseKNN(E, k = 10, method = "cosine", dist = FALSE, ncores = 4)
```

Note that `1 - cosine` is not a metric — the triangle inequality can fail. Use
`acos(similarity) / pi` if a true angular metric is needed.

## Performance

- The Jaccard, cosine, Euclidean and Manhattan kernels traverse only stored
  non-zero entries, so pairwise cost depends on sparsity rather than on the
  full matrix dimensions. Pearson and covariance densify each column, and
  Jensen-Shannon takes dense input.
- Kernels are parallelised with OpenMP where the compiler supports it, and
  report progress via `RcppProgress`. `ncores` defaults to 1; pass `0` to
  auto-detect.
- Peak memory for the full matrix is `O(ncol²)`. Use `sparseKNN()` for problems
  that need to stay feasible.

### macOS note

Apple's clang ships without OpenMP, so a default macOS build runs
single-threaded and `ncores` has no effect. To enable it, install `libomp`
(`brew install libomp`) and add to `~/.R/Makevars`:

```make
LIBOMP = /opt/homebrew/opt/libomp     # /usr/local/opt/libomp on Intel Macs
CPPFLAGS += -Xclang -fopenmp -I$(LIBOMP)/include
LDFLAGS  += -L$(LIBOMP)/lib -lomp
```

## License

GPL-3
