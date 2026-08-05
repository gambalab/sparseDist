// =============================================================================
//  snn.cpp -- Jaccard coefficients between k-nearest-neighbour sets.
//
//  Given the neighbour index matrix produced by sparseKNN(), this computes the
//  shared-nearest-neighbour (SNN) weight of each edge:
//
//      w(i, j) = |N(i) INTERSECT N(j)| / |N(i) UNION N(j)|
//
//  for every j listed among i's neighbours. That is the standard second stage
//  of graph-based clustering: a kNN graph is reweighted by neighbourhood
//  overlap, then partitioned by a community-detection algorithm.
//
//  Adapted from the RcppParallel implementation by Gennaro Gambardella
//  (12/08/2019), itself following
//  https://github.com/tnagler/RcppThread/commit/c26fc2b0d56555fa434c33352747822691334fe8
//
// -----------------------------------------------------------------------------
//  DIFFERENCES FROM THE ORIGINAL
// -----------------------------------------------------------------------------
//  * OpenMP instead of RcppParallel. The rest of the package already uses
//    OpenMP, so this avoids a second dependency and, more importantly, avoids
//    running two thread pools (TBB and OpenMP) in one shared object, which
//    oversubscribes the machine because neither pool knows about the other.
//
//  * Neighbour sets are built ONCE, up front. The original copied and re-sorted
//    row i inside the inner loop, so each row was sorted k times instead of
//    once -- a factor of k of redundant work.
//
//  * Set sizes are used for the union. The original assumed |N(i)| = |N(j)| = k
//    and computed the union as 2k - |intersection|, which is wrong whenever a
//    row holds duplicate or missing entries.
//
//  * Indices are validated. The original used mat(i, j) - 1 directly as a row
//    index with no bounds check, so a malformed entry was an out-of-bounds read
//    (segfault or silent corruption). Entries are now checked before use, with
//    NA accepted as padding.
//
//  * The intersection is counted by a merge walk rather than materialised into
//    a std::vector whose size was the only thing ever read.
//
//  * Output is a weight matrix aligned with the input index matrix, rather than
//    a padded (n*k) x 3 edge list in which unused rows had to be recognised by
//    a (0, 0, 0) sentinel. sparseSNN() turns it into a sparse adjacency matrix.
// =============================================================================

#include <Rcpp.h>
#include <progress.hpp>

#include <vector>
#include <algorithm>
#include <cstddef>

#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::plugins(openmp)]]
// [[Rcpp::depends(RcppProgress)]]

namespace {

// |a INTERSECT b| for two ascending, de-duplicated sequences.
inline std::size_t inter_size(const std::vector<int>& a, const std::vector<int>& b) {
  std::size_t c = 0, ia = 0, ib = 0;
  const std::size_t na = a.size(), nb = b.size();
  while (ia < na && ib < nb) {
    if (a[ia] < b[ib])      ++ia;
    else if (b[ib] < a[ia]) ++ib;
    else { ++c; ++ia; ++ib; }
  }
  return c;
}

// Raise an R error if the user interrupted during the parallel region. Must be
// called AFTER the region: an exception may never leave an OpenMP block.
inline void stop_if_aborted(const Progress& p) {
  if (p.is_aborted()) Rcpp::stop("Computation interrupted by the user.");
}

}  // namespace

// Jaccard weights between neighbour sets.
//
//   idx           n x k matrix of 1-based neighbour indices (NA = padding).
//   include_self  whether i belongs to its own neighbour set. Self entries
//                 already present in a row are removed first, so this decides
//                 membership outright rather than just appending.
//   ncores        threads.
//   verbose       draw a progress bar.
//
// Returns an n x k numeric matrix aligned with idx: entry (i, t) is the
// Jaccard overlap between the neighbourhoods of i and idx(i, t), or NA where
// idx is NA or a neighbourhood is empty.
//
// Not exported to users; called from sparseSNN().
// [[Rcpp::export]]
Rcpp::NumericMatrix snnJaccard(const Rcpp::IntegerMatrix& idx,
                               bool include_self = true,
                               int ncores = 1,
                               bool verbose = false) {
  const int n = idx.nrow();
  const int k = idx.ncol();
  if (n < 1 || k < 1) Rcpp::stop("'idx' must have at least one row and column.");
  if (ncores < 1) ncores = 1;

  const std::size_t N = static_cast<std::size_t>(n);
  const std::size_t K = static_cast<std::size_t>(k);
  const std::size_t total = N * K;

  // Validate and copy into a plain buffer BEFORE any parallel region: R/Rcpp
  // objects must not be touched from worker threads, and an error may only be
  // raised on the main thread. Same pattern as topKBlock().
  std::vector<int> idx_buf(total);
  for (int t = 0; t < k; ++t) {
    for (int i = 0; i < n; ++i) {
      const int v = idx(i, t);
      if (v != NA_INTEGER && (v < 1 || v > n)) {
        Rcpp::stop("'idx' entries must be NA or 1-based row indices in "
                   "[1, nrow(idx)].");
      }
      idx_buf[static_cast<std::size_t>(i) + N * static_cast<std::size_t>(t)] = v;
    }
  }

  // column-major accessor over the copied buffer
  const int* ib = idx_buf.data();
  #define IDX_AT(i, t) ib[static_cast<std::size_t>(i) + N * static_cast<std::size_t>(t)]

  // Build every neighbour set once, ascending and de-duplicated. The outer
  // vector is fully allocated before the parallel region and each iteration
  // touches a distinct inner vector, so this is safe.
  std::vector< std::vector<int> > sets(N);
  const long long n_ll = static_cast<long long>(n);

#pragma omp parallel for num_threads(ncores) schedule(static)
  for (long long ii = 0; ii < n_ll; ii++) {
    const int i = static_cast<int>(ii);
    std::vector<int>& S = sets[static_cast<std::size_t>(i)];
    S.reserve(K + 1u);
    // include_self decides MEMBERSHIP, not merely whether to append: any self
    // index already present in the row is dropped first, so the neighbourhood
    // is the same whether or not the kNN search returned the point itself.
    if (include_self) S.push_back(i);
    for (int t = 0; t < k; ++t) {
      const int v = IDX_AT(i, t);
      if (v == NA_INTEGER) continue;
      const int neighbour = v - 1;               // -> 0-based
      if (neighbour != i) S.push_back(neighbour);
    }
    std::sort(S.begin(), S.end());
    S.erase(std::unique(S.begin(), S.end()), S.end());
  }

  std::vector<double> out(total, NA_REAL);

  // One tick per row. The cost is roughly O(n * k^2), so a large graph can run
  // long enough that interrupt responsiveness matters.
  Progress p(static_cast<unsigned long>(n), verbose);

#pragma omp parallel for num_threads(ncores) schedule(dynamic)
  for (long long ii = 0; ii < n_ll; ii++) {
    if (Progress::check_abort()) continue;
    const int i = static_cast<int>(ii);
    const std::vector<int>& Si = sets[static_cast<std::size_t>(i)];
    if (Si.empty()) { p.increment(); continue; }  // no usable neighbourhood

    for (int t = 0; t < k; ++t) {
      const int v = IDX_AT(i, t);
      if (v == NA_INTEGER) continue;             // stays NA
      const std::vector<int>& Sj = sets[static_cast<std::size_t>(v - 1)];
      if (Sj.empty()) continue;

      const std::size_t u = inter_size(Si, Sj);
      // |A UNION B| = |A| + |B| - |A INTERSECT B|, >= 1 since neither is empty
      const double denom = static_cast<double>(Si.size() + Sj.size() - u);
      out[static_cast<std::size_t>(i) + N * static_cast<std::size_t>(t)] =
          static_cast<double>(u) / denom;
    }
    p.increment();
  }

  #undef IDX_AT

  stop_if_aborted(p);

  // Build the R object only after every thread has finished.
  Rcpp::NumericMatrix w(n, k);
  std::copy(out.begin(), out.end(), w.begin());
  return w;
}
