// =============================================================================
//  topK.cpp -- reduce a dense distance/similarity block to its k best entries
//              per query column.
//
//  This is the only new kernel needed for blocked k-nearest-neighbour search.
//  The distance blocks themselves are produced by the existing two-matrix
//  kernels in sparseDist.cpp, which are left untouched: a call such as
//
//      D <- fastEuclidean2(m = reference, m2 = queryBlock)
//
//  already yields exactly the ncol(reference) x ncol(queryBlock) block that
//  this function consumes. Blocking is therefore driven from R (see
//  sparseKNN()), and peak memory is O(ncol(reference) * block_size) rather
//  than O(ncol^2).
//
//  Selection uses a bounded heap of size k: O(n log k) per column with O(k)
//  extra memory, instead of sorting the whole column. Ties are broken by the
//  lower row index, so the result does not depend on the number of threads.
// =============================================================================

#include <RcppArmadillo.h>

#include <algorithm>
#include <vector>
#include <utility>
#include <cmath>
#include <cstddef>

#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

namespace {

typedef std::pair<double, arma::uword> Cand;   // (value, 0-based row index)

// Strict weak ordering: "a is a better neighbour than b".
// Used as the heap comparator, so the heap top is the WORST kept candidate.
struct Better {
  bool decreasing;
  explicit Better(bool d) : decreasing(d) {}
  bool operator()(const Cand& a, const Cand& b) const {
    if (a.first != b.first) return decreasing ? (a.first > b.first) : (a.first < b.first);
    return a.second < b.second;               // deterministic tie-break
  }
};

}  // namespace

// Reduce a distance block to the k best entries per column.
//
//   D          dense block; D(i, b) relates reference column i to query col b.
//   k          number of neighbours to keep.
//   decreasing keep the k LARGEST values (similarity) rather than the smallest.
//   self_row   for each query column, the 1-based row of D holding its own
//              self-comparison, or 0 when there is none; those are excluded.
//   ncores     threads.
//
// Returns list(idx, dist): ncol(D) x k matrices, idx holding 1-based row
// indices into D. Slots with no candidate are NA (not NaN).
//
// Not exported to users; called from sparseKNN().
// [[Rcpp::export]]
Rcpp::List topKBlock(const arma::mat& D, int k, bool decreasing,
                     const Rcpp::IntegerVector& self_row, int ncores = 1) {
  const arma::uword nrow = D.n_rows;
  const arma::uword ncol = D.n_cols;

  if (k < 1) Rcpp::stop("k must be >= 1.");
  if (static_cast<arma::uword>(self_row.size()) != ncol) {
    Rcpp::stop("self_row must have one entry per column of D.");
  }
  if (ncores < 1) ncores = 1;

  const arma::uword K = static_cast<arma::uword>(k);
  const Better cmp(decreasing);

  // Copy self_row into a plain buffer BEFORE the parallel region: R/Rcpp
  // objects should not be touched from worker threads. Validating here also
  // means the hot loop can trust the values.
  std::vector<int> self(static_cast<std::size_t>(ncol));
  for (arma::uword b = 0; b < ncol; ++b) {
    const int sr = self_row[b];
    if (sr == NA_INTEGER || sr < 0 ||
        static_cast<arma::uword>(sr) > nrow) {
      Rcpp::stop("self_row entries must be 0 or valid 1-based rows of D.");
    }
    self[static_cast<std::size_t>(b)] = sr;
  }

  // Plain buffers: written from several threads, so no R allocation happens
  // inside the parallel region. Converted to R matrices afterwards.
  // NA_REAL, not quiet_NaN(): an unfilled slot is a MISSING neighbour, and R
  // distinguishes NA from NaN. Every accepted candidate is finite, so NA can
  // never collide with a real value. Indices use 0 as the "empty" sentinel
  // because valid ones are 1-based.
  std::vector<double> out_val(static_cast<std::size_t>(ncol) * K, NA_REAL);
  std::vector<double> out_idx(static_cast<std::size_t>(ncol) * K, 0.0);

  const long long ncol_ll = static_cast<long long>(ncol);

#pragma omp parallel for num_threads(ncores) schedule(dynamic)
  for (long long bb = 0; bb < ncol_ll; bb++) {
    const arma::uword b = static_cast<arma::uword>(bb);

    // nrow is an impossible row index, so it means "exclude nothing".
    const int sr = self[static_cast<std::size_t>(b)];
    const arma::uword skip = (sr > 0) ? static_cast<arma::uword>(sr - 1) : nrow;

    std::vector<Cand> heap;
    heap.reserve(K + 1);

    for (arma::uword i = 0; i < nrow; ++i) {
      if (i == skip) continue;
      const double v = D(i, b);
      if (!std::isfinite(v)) continue;        // never let NA/NaN win a slot

      if (heap.size() < K) {
        heap.push_back(Cand(v, i));
        std::push_heap(heap.begin(), heap.end(), cmp);
      } else if (cmp(Cand(v, i), heap.front())) {
        std::pop_heap(heap.begin(), heap.end(), cmp);
        heap.back() = Cand(v, i);
        std::push_heap(heap.begin(), heap.end(), cmp);
      }
    }

    std::sort_heap(heap.begin(), heap.end(), cmp);   // best first

    for (std::size_t t = 0; t < heap.size(); ++t) {
      const std::size_t at = static_cast<std::size_t>(t) * ncol + b;  // column-major
      out_val[at] = heap[t].first;
      out_idx[at] = static_cast<double>(heap[t].second + 1);          // 1-based
    }
  }

  Rcpp::NumericMatrix dist(ncol, K);
  Rcpp::IntegerMatrix idx(ncol, K);
  for (std::size_t at = 0; at < out_val.size(); ++at) {
    dist[at] = out_val[at];
    idx[at]  = (out_idx[at] > 0.0) ? static_cast<int>(out_idx[at]) : NA_INTEGER;
  }

  return Rcpp::List::create(Rcpp::Named("idx")  = idx,
                            Rcpp::Named("dist") = dist);
}
