// =============================================================================
//  snn.cpp -- Jaccard coefficients between k-nearest-neighbour sets.
//
//  Given the neighbour index matrix produced by sparseKNN(), this computes the
//  shared-nearest-neighbour (SNN) weight of an edge:
//
//      w(i, j) = |N(i) INTERSECT N(j)| / |N(i) UNION N(j)|
//
//  Two edge sets are supported, and they are genuinely different graphs:
//
//    snnJaccard()        w for every j listed among i's neighbours. The edge
//                        set is the kNN graph itself.
//
//    snnJaccardShared()  w for every pair sharing at least one neighbour,
//                        whether or not they are kNN-adjacent. This is the
//                        convention used by Seurat and by bluster's
//                        neighborsToSNNGraph(type = "jaccard").
//
//  The first is a subset of the second -- usually proper, and substantially
//  sparser -- and where both emit an edge the weights are identical; only the
//  edge set differs. The second runs to roughly n * k^2 edges against n * k.
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

// Validate idx and copy it into a plain column-major buffer.
//
// Done BEFORE any parallel region: R/Rcpp objects must not be touched from
// worker threads, and an error may only be raised on the main thread.
inline void copy_indices(const Rcpp::IntegerMatrix& idx, int n, int k,
                         std::vector<int>& buf) {
  const std::size_t N = static_cast<std::size_t>(n);
  buf.resize(N * static_cast<std::size_t>(k));
  for (int t = 0; t < k; ++t) {
    for (int i = 0; i < n; ++i) {
      const int v = idx(i, t);
      if (v != NA_INTEGER && (v < 1 || v > n)) {
        Rcpp::stop("'idx' entries must be NA or 1-based row indices in "
                   "[1, nrow(idx)].");
      }
      buf[static_cast<std::size_t>(i) + N * static_cast<std::size_t>(t)] = v;
    }
  }
}

inline int idx_at(const int* ib, std::size_t N, int i, int t) {
  return ib[static_cast<std::size_t>(i) + N * static_cast<std::size_t>(t)];
}

// Build every neighbour set once: ascending, de-duplicated, 0-based.
//
// include_self decides MEMBERSHIP, not merely whether to append: any self index
// already present in the row is dropped first, so the neighbourhood is the same
// whether or not the kNN search returned the point itself.
inline void build_sets(const int* ib, std::size_t N, int n, int k,
                       bool include_self, int ncores,
                       std::vector< std::vector<int> >& sets) {
  sets.assign(N, std::vector<int>());
  const std::size_t K = static_cast<std::size_t>(k);
  const long long n_ll = static_cast<long long>(n);

#pragma omp parallel for num_threads(ncores) schedule(static)
  for (long long ii = 0; ii < n_ll; ii++) {
    const int i = static_cast<int>(ii);
    std::vector<int>& S = sets[static_cast<std::size_t>(i)];
    S.reserve(K + 1u);
    if (include_self) S.push_back(i);
    for (int t = 0; t < k; ++t) {
      const int v = idx_at(ib, N, i, t);
      if (v == NA_INTEGER) continue;
      const int neighbour = v - 1;               // -> 0-based
      if (neighbour != i) S.push_back(neighbour);
    }
    std::sort(S.begin(), S.end());
    S.erase(std::unique(S.begin(), S.end()), S.end());
  }
}

// Tally |N(i) INTERSECT N(j)| into `acc` for every j > i sharing a neighbour
// with i, recording which slots were touched.
//
// SHARED BY BOTH PASSES of snnJaccardShared(), deliberately. The two-pass
// design is only sound if the passes agree exactly on which edges survive, and
// two hand-maintained copies of this loop would eventually drift.
//
// Each reverse list is ASCENDING -- it is filled while i advances monotonically
// -- so upper_bound() jumps straight to the first j > i. Testing `j <= i` inside
// the loop instead would still walk every entry, halving the accumulator
// updates but not the traversal, leaving the cost at sum_m |R(m)|^2 rather than
// half of it. For very short reverse lists the binary search is not free; a
// linear skip may profile better, but this is the clean formulation.
inline void accumulate_row(int i,
                           const std::vector<int>& Si,
                           const std::vector<std::size_t>& rev_ptr,
                           const std::vector<int>& rev_idx,
                           std::vector<int>& acc,
                           std::vector<int>& touched) {
  for (std::size_t a = 0; a < Si.size(); ++a) {
    const std::size_t m = static_cast<std::size_t>(Si[a]);
    const std::vector<int>::const_iterator lo =
        rev_idx.begin() + static_cast<std::ptrdiff_t>(rev_ptr[m]);
    const std::vector<int>::const_iterator hi =
        rev_idx.begin() + static_cast<std::ptrdiff_t>(rev_ptr[m + 1u]);
    for (std::vector<int>::const_iterator q = std::upper_bound(lo, hi, i);
         q != hi; ++q) {
      const int j = *q;
      if (acc[static_cast<std::size_t>(j)]++ == 0) touched.push_back(j);
    }
  }
}

// |A INTERSECT B| / |A UNION B| from the overlap count and the two set sizes.
// The union is computed from the ACTUAL cardinalities rather than assumed to be
// 2k - overlap, which is only correct when both sets hold exactly k members.
inline double jaccard_from_counts(std::size_t u, std::size_t ni, std::size_t nj) {
  return static_cast<double>(u) / static_cast<double>(ni + nj - u);
}

}  // namespace

// Jaccard weights over the kNN edge set.
//
//   idx           n x k matrix of 1-based neighbour indices (NA = padding).
//   include_self  whether i belongs to its own neighbour set.
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
  const std::size_t total = N * static_cast<std::size_t>(k);

  std::vector<int> idx_buf;
  copy_indices(idx, n, k, idx_buf);
  const int* ib = idx_buf.data();

  std::vector< std::vector<int> > sets;
  build_sets(ib, N, n, k, include_self, ncores, sets);

  std::vector<double> out(total, NA_REAL);

  // One tick per row. The cost is roughly O(n * k^2), so a large graph can run
  // long enough that interrupt responsiveness matters.
  Progress p(static_cast<unsigned long>(N), verbose);
  const long long n_ll = static_cast<long long>(n);

#pragma omp parallel for num_threads(ncores) schedule(dynamic)
  for (long long ii = 0; ii < n_ll; ii++) {
    if (Progress::check_abort()) continue;
    const int i = static_cast<int>(ii);
    const std::vector<int>& Si = sets[static_cast<std::size_t>(i)];
    if (Si.empty()) { p.increment(); continue; }  // no usable neighbourhood

    for (int t = 0; t < k; ++t) {
      const int v = idx_at(ib, N, i, t);
      if (v == NA_INTEGER) continue;             // stays NA
      const std::vector<int>& Sj = sets[static_cast<std::size_t>(v - 1)];
      if (Sj.empty()) continue;

      const std::size_t u = inter_size(Si, Sj);
      out[static_cast<std::size_t>(i) + N * static_cast<std::size_t>(t)] =
          jaccard_from_counts(u, Si.size(), Sj.size());
    }
    p.increment();
  }

  stop_if_aborted(p);

  // Build the R object only after every thread has finished.
  Rcpp::NumericMatrix w(n, k);
  std::copy(out.begin(), out.end(), w.begin());
  return w;
}

// -----------------------------------------------------------------------------
//  Shared-neighbour edge set (Seurat / bluster convention)
// -----------------------------------------------------------------------------
//
//  Emits w(i, j) for EVERY pair sharing at least one neighbour, not only for
//  kNN-adjacent pairs. The candidate set is therefore not known in advance and
//  has to be discovered, which calls for a different algorithm from
//  snnJaccard(): an inverted index plus a sparse accumulator, i.e. exactly the
//  row-wise traversal used for a sparse matrix product A * t(A), where A is the
//  binary neighbourhood-membership matrix.
//
//  For each i, walk m in N(i) and then j in R(m) = {j : m in N(j)}, tallying
//  |N(i) INTERSECT N(j)| in a scratch array indexed by j. Work is bounded by
//  sum_m |R(m)|^2, the same as the equivalent matrix product, and roughly half
//  of that given the upper-triangle skip -- but unlike CHOLMOD's product it
//  parallelises cleanly over i.
//
//  DESIGN NOTES
//
//  * UPPER TRIANGLE ONLY. The weight is symmetric and, unlike the kNN edge set,
//    the shared-neighbour edge set is symmetric too -- so there is no directed
//    variant to preserve. accumulate_row() binary-searches into each reverse
//    list rather than filtering as it goes, so the traversal really is halved
//    and not merely the updates. sparseSNN() mirrors the triangle when
//    assembling the matrix.
//
//  * PRUNING HAPPENS HERE, not in R. This mode produces on the order of n * k^2
//    edges (roughly k times the kNN graph), and Seurat's customary
//    prune.SNN = 1/15 discards most of them. Filtering inside the kernel means
//    the dropped edges are never materialised at all.
//
//  * TWO PASSES, EXACT ALLOCATION. Pass one counts surviving edges per row;
//    a prefix sum then gives each row a private slice of the output, which pass
//    two fills without any locking or concatenation. This costs roughly twice
//    the accumulation work but peaks at exactly the output size. The
//    alternative -- per-thread buffers merged afterwards -- runs one pass but
//    peaks at about twice the output, and a bounded peak is the property this
//    package exists to provide. Revisit if profiling shows pass one dominating.
//
//  * SCRATCH is ncores * n ints for the accumulator plus a touched-list. At
//    n = 1e6 and 32 threads that is on the order of 250 MB, which is worth
//    knowing about before it is discovered.
//
//  Returns a triplet list (i, j, x) of 1-based upper-triangle edges.
//
//  Not exported to users; called from sparseSNN().
// [[Rcpp::export]]
Rcpp::List snnJaccardShared(const Rcpp::IntegerMatrix& idx,
                            bool include_self = true,
                            double prune = 0.0,
                            int ncores = 1,
                            bool verbose = false) {
  const int n = idx.nrow();
  const int k = idx.ncol();
  if (n < 1 || k < 1) Rcpp::stop("'idx' must have at least one row and column.");
  if (ncores < 1) ncores = 1;

  const std::size_t N = static_cast<std::size_t>(n);

  std::vector<int> idx_buf;
  copy_indices(idx, n, k, idx_buf);
  const int* ib = idx_buf.data();

  std::vector< std::vector<int> > sets;
  build_sets(ib, N, n, k, include_self, ncores, sets);

  // ---- inverted index: R(m) = { i : m in N(i) }, as CSR -------------------
  // Counting sort in two sweeps. Serial, but only O(sum |N(i)|) = O(n * k),
  // negligible beside the accumulation that follows. Because i advances
  // monotonically while filling, each slice comes out ASCENDING -- which is
  // what lets accumulate_row() binary-search past the lower triangle.
  std::vector<std::size_t> rev_ptr(N + 1u, 0);
  for (std::size_t i = 0; i < N; ++i) {
    const std::vector<int>& S = sets[i];
    for (std::size_t a = 0; a < S.size(); ++a) {
      ++rev_ptr[static_cast<std::size_t>(S[a]) + 1u];
    }
  }
  for (std::size_t m = 0; m < N; ++m) rev_ptr[m + 1u] += rev_ptr[m];

  std::vector<int> rev_idx(rev_ptr[N]);
  {
    std::vector<std::size_t> fill(rev_ptr.begin(), rev_ptr.end() - 1);
    for (std::size_t i = 0; i < N; ++i) {
      const std::vector<int>& S = sets[i];
      for (std::size_t a = 0; a < S.size(); ++a) {
        rev_idx[fill[static_cast<std::size_t>(S[a])]++] = static_cast<int>(i);
      }
    }
  }

  // Two ticks per row, one per pass. Widened before multiplying: 2 * n would
  // evaluate as int and only then be cast, which contradicts the overflow
  // discipline the distance kernels follow even though it is unreachable here.
  const unsigned long progress_total =
      static_cast<unsigned long>(2ULL * static_cast<unsigned long long>(N));
  Progress p(progress_total, verbose);
  const long long n_ll = static_cast<long long>(n);

  // ---- pass 1: count surviving edges per row ------------------------------
  std::vector<std::size_t> row_cnt(N, 0);

#pragma omp parallel num_threads(ncores)
  {
    std::vector<int> acc(N, 0);     // per-thread sparse accumulator
    std::vector<int> touched;
    touched.reserve(1024);

#pragma omp for schedule(dynamic, 64)
    for (long long ii = 0; ii < n_ll; ii++) {
      if (Progress::check_abort()) continue;
      const int i = static_cast<int>(ii);
      const std::vector<int>& Si = sets[static_cast<std::size_t>(i)];
      if (Si.empty()) { p.increment(); continue; }

      accumulate_row(i, Si, rev_ptr, rev_idx, acc, touched);

      std::size_t kept = 0;
      for (std::size_t a = 0; a < touched.size(); ++a) {
        const std::size_t j = static_cast<std::size_t>(touched[a]);
        const std::size_t u = static_cast<std::size_t>(acc[j]);
        if (jaccard_from_counts(u, Si.size(), sets[j].size()) > prune) ++kept;
        acc[j] = 0;
      }
      touched.clear();
      row_cnt[static_cast<std::size_t>(i)] = kept;
      p.increment();
    }
  }

  stop_if_aborted(p);

  // ---- prefix sum: a private output slice per row -------------------------
  std::vector<std::size_t> offset(N + 1u, 0);
  for (std::size_t i = 0; i < N; ++i) offset[i + 1u] = offset[i] + row_cnt[i];
  const std::size_t n_edges = offset[N];

  Rcpp::IntegerVector out_i(n_edges);
  Rcpp::IntegerVector out_j(n_edges);
  Rcpp::NumericVector out_x(n_edges);

  if (n_edges == 0u) {
    // Account for the pass that is about to be skipped, or a verbose run
    // (prune = 1 being the obvious case) would leave the bar stuck at half.
    p.increment(static_cast<unsigned long>(N));
    return Rcpp::List::create(Rcpp::Named("i") = out_i,
                              Rcpp::Named("j") = out_j,
                              Rcpp::Named("x") = out_x);
  }

  int* pi = out_i.begin();
  int* pj = out_j.begin();
  double* px = out_x.begin();

  // ---- pass 2: fill, each row writing only into its own slice -------------
  // Same traversal helper and same arithmetic as pass 1, so the counts agree
  // exactly; no locking is needed because the slices are disjoint.
#pragma omp parallel num_threads(ncores)
  {
    std::vector<int> acc(N, 0);
    std::vector<int> touched;
    touched.reserve(1024);

#pragma omp for schedule(dynamic, 64)
    for (long long ii = 0; ii < n_ll; ii++) {
      if (Progress::check_abort()) continue;
      const int i = static_cast<int>(ii);
      const std::vector<int>& Si = sets[static_cast<std::size_t>(i)];
      if (Si.empty()) { p.increment(); continue; }

      accumulate_row(i, Si, rev_ptr, rev_idx, acc, touched);

      std::size_t at = offset[static_cast<std::size_t>(i)];
      for (std::size_t a = 0; a < touched.size(); ++a) {
        const std::size_t j = static_cast<std::size_t>(touched[a]);
        const std::size_t u = static_cast<std::size_t>(acc[j]);
        const double w = jaccard_from_counts(u, Si.size(), sets[j].size());
        if (w > prune) {
          pi[at] = i + 1;                        // -> 1-based for R
          pj[at] = static_cast<int>(j) + 1;
          px[at] = w;
          ++at;
        }
        acc[j] = 0;
      }
      touched.clear();
      p.increment();
    }
  }

  stop_if_aborted(p);

  return Rcpp::List::create(Rcpp::Named("i") = out_i,
                            Rcpp::Named("j") = out_j,
                            Rcpp::Named("x") = out_x);
}
