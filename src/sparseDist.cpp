// =============================================================================
//  sparseDist.cpp
// =============================================================================
//
//  Column-wise distance / similarity kernels over Armadillo sparse (and, for
//  Jensen-Shannon, dense) matrices, parallelised with OpenMP and reporting
//  progress via RcppProgress.
//
//  KNOWN LIMITATION
//    * Peak memory is O(n_cols^2): the all-pairs result is materialised in full
//      before being returned. That is inherent to returning the whole matrix;
//      only a blocked or streaming API (emitting pairs above a threshold) would
//      avoid it.
// =============================================================================

#include <RcppArmadillo.h>
#include <progress.hpp>
#include <cmath>
#include <string>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]
// [[Rcpp::depends(RcppProgress)]]

// ---------------------------------------------------------------------------
//  file-local helpers (not exported to R)
// ---------------------------------------------------------------------------

// num_threads(<1) is undefined behaviour; clamp to a sane minimum.
static inline int sanitize_ncores(int ncores) {
  return (ncores < 1) ? 1 : ncores;
}

// If the user interrupted during the OpenMP computation, fail loudly rather than
// return a half-filled matrix. p.is_aborted() reports whether an interrupt was
// detected *during* the parallel region (the master thread sets the flag inside
// check_abort(); it is reset when each Progress is constructed) -- this is the
// precise question "did this computation abort?", and unlike a fresh check it
// won't discard an already-completed result if a Ctrl-C lands during the final
// barrier (that pending interrupt is still raised at R's next checkpoint). MUST
// be called AFTER the parallel region -- an exception must never leave an OpenMP
// block. No separate std::atomic is needed; RcppProgress owns the flag.
static inline void stop_if_aborted(const Progress& p) {
  if (p.is_aborted()) Rcpp::stop("Computation interrupted by the user.");
}

// Exact number of (i,j) pairs visited by the triangular loops, so the progress
// total always matches the number of increments (independent of `full`).
//   diag=true : n + (n-1) + ... + 1 = n(n+1)/2
//   diag=false: (n-1) + ... + 0     = n(n-1)/2
static inline unsigned long long tri_pairs(arma::uword n, bool diag) {
  const unsigned long long N = static_cast<unsigned long long>(n);
  if (N == 0ULL) return 0ULL;
  return diag ? (N * (N + 1ULL)) / 2ULL : (N * (N - 1ULL)) / 2ULL;
}

// The pair counts above are exact 64-bit values, but RcppProgress stores its
// counter as `unsigned long` -- only 32-bit on LLP64 platforms (e.g. Windows).
// Saturate at ULONG_MAX so a huge total can't wrap to a small number and make
// the progress *display* jump past 100%. (Results are unaffected: loop indices
// are 64-bit. This bound is unreachable in practice -- the dense n_cols x n_cols
// result would need astronomically more memory than the counter's range.)
static inline unsigned long clamp_progress(unsigned long long total) {
  const unsigned long long max_ul =
      static_cast<unsigned long long>(std::numeric_limits<unsigned long>::max());
  return static_cast<unsigned long>(total < max_ul ? total : max_ul);
}

// x * log(x) with the convention 0*log(0) = 0 (used by Jensen-Shannon).
// fastJS/fastJS2 validate that inputs are finite and non-negative, so the only
// non-finite term produced here is 0*log(0) = NaN, which is mapped to 0.
static inline arma::vec xlogx(const arma::vec& x) {
  arma::vec r = x % arma::log(x);         // 0 * (-inf) -> NaN
  r.replace(arma::datum::nan, 0.0);       // ...treated as 0
  return r;
}

// ---------------------------------------------------------------------------
//  Correlation
// ---------------------------------------------------------------------------

// Correlation (or 1-correlation distance) between the columns of a sparse matrix m.
// [[Rcpp::export]]
SEXP fastCorr(const arma::sp_mat& m, int ncores=1, bool verbose=true,
                      bool full=false, bool diag=true, bool dist=true)
{
  ncores = sanitize_ncores(ncores);
  const arma::uword ncol = m.n_cols;
  const long long   ncol_ll = static_cast<long long>(ncol);
  arma::mat d(ncol, ncol, arma::fill::zeros);

  Progress p(clamp_progress(tri_pairs(ncol, diag)), verbose);

#pragma omp parallel for num_threads(ncores) shared(d) schedule(dynamic)
  for (long long ii = 0; ii < ncol_ll; ii++) {
    if (Progress::check_abort()) continue;
    const arma::uword i = static_cast<arma::uword>(ii);
    const arma::vec li = arma::vec(m.col(i));
    unsigned long long steps = 0;

    for (arma::uword j = (diag ? i : i + 1); j < ncol; j++) {
      double val;
      if (i == j) {
        // Self-comparison: a column is perfectly correlated with itself, so its
        // distance to itself is exactly 0 (similarity 1) -- even for a degenerate
        // column where cor() is undefined. Keeps the distance diagonal at 0.
        val = dist ? 0.0 : 1.0;
      } else {
        // norm_type 0 (= N-1) is Armadillo's default and is passed explicitly in
        // both fastCorr and fastCorr2 so the two agree on sight. The choice is
        // immaterial for CORRELATION -- the N vs N-1 factor cancels in
        // cov/(sd*sd) -- but it does matter for cov(), which uses the same default.
        double c = arma::as_scalar(arma::cor(li, arma::vec(m.col(j)), 0));
        if (!std::isfinite(c)) c = 0.0;    // undefined (constant/empty column) -> 0
        // arma::cor can return a value a few ULP outside [-1, 1] for
        // near-identical columns; clamping keeps the distance in [0, 2] and
        // stops an identical pair producing a tiny NEGATIVE distance.
        if (c >  1.0) c =  1.0;
        if (c < -1.0) c = -1.0;
        val = dist ? (1.0 - c) : c;
      }
      d(j, i) = val;
      if (full) d(i, j) = val;
      ++steps;
    }
    p.increment(static_cast<unsigned long>(steps));
  }

  stop_if_aborted(p);

  // The loop above only visits j == i when diag=true, so in SIMILARITY mode
  // (dist=false) the diagonal would otherwise be left at the zero-fill -- i.e.
  // a column would report 0 similarity with itself. The self-similarity is 1 by
  // definition, so write it unconditionally. (In DISTANCE mode the correct
  // self-distance is 0, which the zero-fill already gives, so nothing to do.)
  // Note this makes `diag` affect only how much work the loop does, not the
  // value on the diagonal, which is now always mathematically correct.
  if (!dist) d.diag().ones();

  // Return type follows the data (see RESULT STORAGE in the header):
  // the distance form is non-zero for almost every column pair -> dense;
  // the similarity form is exactly 0 for disjoint pairs -> sparse.
  if (dist) return Rcpp::wrap(d);
  return Rcpp::wrap(arma::sp_mat(d));
}

// [[Rcpp::export]]
SEXP fastCorr2(const arma::sp_mat& m, const arma::sp_mat& m2,
                       int ncores=1, bool verbose=true, bool dist=true)
{
  if (m.n_rows != m2.n_rows) {
    Rcpp::stop("Mismatched row dimensions of m (" +
               std::to_string(static_cast<unsigned long long>(m.n_rows)) + ") and m2 (" +
               std::to_string(static_cast<unsigned long long>(m2.n_rows)) + ").");
  }
  ncores = sanitize_ncores(ncores);
  const arma::uword nc = m.n_cols, nc2 = m2.n_cols;
  const long long   nc_ll = static_cast<long long>(nc);
  arma::mat d(nc, nc2, arma::fill::zeros);

  Progress p(clamp_progress(static_cast<unsigned long long>(nc) * static_cast<unsigned long long>(nc2)), verbose);

#pragma omp parallel for num_threads(ncores) shared(d) schedule(dynamic)
  for (long long ii = 0; ii < nc_ll; ii++) {
    if (Progress::check_abort()) continue;
    const arma::uword i = static_cast<arma::uword>(ii);
    const arma::vec li = arma::vec(m.col(i));    // densified once per i

    for (arma::uword j = 0; j < nc2; j++) {
      double c = arma::as_scalar(arma::cor(li, arma::vec(m2.col(j)), 0));
      if (!std::isfinite(c)) c = 0.0;
      // See fastCorr: keep c in [-1, 1] so the distance cannot go negative.
      if (c >  1.0) c =  1.0;
      if (c < -1.0) c = -1.0;
      d(i, j) = dist ? (1.0 - c) : c;
    }
    p.increment(static_cast<unsigned long>(nc2));
  }

  stop_if_aborted(p);
  // Return type follows the data (see RESULT STORAGE in the header):
  // the distance form is non-zero for almost every column pair -> dense;
  // the similarity form is exactly 0 for disjoint pairs -> sparse.
  if (dist) return Rcpp::wrap(d);
  return Rcpp::wrap(arma::sp_mat(d));
}

// ---------------------------------------------------------------------------
//  Covariance
// ---------------------------------------------------------------------------

// Covariance between the columns of a sparse matrix m.
// (The unused `dist` argument of the original has been removed.)
// [[Rcpp::export]]
arma::sp_mat fastCov(const arma::sp_mat& m, int ncores=1, bool verbose=true,
                     bool full=false, bool diag=true)
{
  ncores = sanitize_ncores(ncores);
  const arma::uword ncol = m.n_cols;
  const long long   ncol_ll = static_cast<long long>(ncol);
  arma::mat d(ncol, ncol, arma::fill::zeros);

  Progress p(clamp_progress(tri_pairs(ncol, diag)), verbose);

#pragma omp parallel for num_threads(ncores) shared(d) schedule(dynamic)
  for (long long ii = 0; ii < ncol_ll; ii++) {
    if (Progress::check_abort()) continue;
    const arma::uword i = static_cast<arma::uword>(ii);
    const arma::vec li = arma::vec(m.col(i));
    unsigned long long steps = 0;

    for (arma::uword j = (diag ? i : i + 1); j < ncol; j++) {
      const double v = arma::as_scalar(arma::cov(li, arma::vec(m.col(j))));
      d(j, i) = v;
      if (full) d(i, j) = v;
      ++steps;
    }
    p.increment(static_cast<unsigned long>(steps));
  }

  stop_if_aborted(p);
  return arma::sp_mat(d);
}

// [[Rcpp::export]]
arma::sp_mat fastCov2(const arma::sp_mat& m, const arma::sp_mat& m2,
                      int ncores=1, bool verbose=true)
{
  if (m.n_rows != m2.n_rows) {
    Rcpp::stop("Mismatched row dimensions of m (" +
               std::to_string(static_cast<unsigned long long>(m.n_rows)) + ") and m2 (" +
               std::to_string(static_cast<unsigned long long>(m2.n_rows)) + ").");
  }
  ncores = sanitize_ncores(ncores);
  const arma::uword nc = m.n_cols, nc2 = m2.n_cols;
  const long long   nc_ll = static_cast<long long>(nc);
  arma::mat d(nc, nc2, arma::fill::zeros);

  Progress p(clamp_progress(static_cast<unsigned long long>(nc) * static_cast<unsigned long long>(nc2)), verbose);

#pragma omp parallel for num_threads(ncores) shared(d) schedule(dynamic)
  for (long long ii = 0; ii < nc_ll; ii++) {
    if (Progress::check_abort()) continue;
    const arma::uword i = static_cast<arma::uword>(ii);
    const arma::vec li = arma::vec(m.col(i));    // densified once per i

    for (arma::uword j = 0; j < nc2; j++) {
      d(i, j) = arma::as_scalar(arma::cov(li, arma::vec(m2.col(j))));
    }
    p.increment(static_cast<unsigned long>(nc2));
  }

  stop_if_aborted(p);
  return arma::sp_mat(d);
}

// ---------------------------------------------------------------------------
//  Jaccard (binary: operates on the sparsity pattern, values ignored)
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
SEXP fastJacc2(const arma::sp_mat& m, const arma::sp_mat& m2,
                       int ncores=1, bool verbose=true, bool dist=true)
{
  if (m.n_rows != m2.n_rows) {
    Rcpp::stop("Mismatched row dimensions of m (" +
               std::to_string(static_cast<unsigned long long>(m.n_rows)) + ") and m2 (" +
               std::to_string(static_cast<unsigned long long>(m2.n_rows)) + ").");
  }
  typedef arma::sp_mat::const_col_iterator iter;
  ncores = sanitize_ncores(ncores);
  const arma::uword nc = m.n_cols, nc2 = m2.n_cols;
  const long long   nc_ll = static_cast<long long>(nc);
  arma::mat d(nc, nc2, arma::fill::zeros);

  Progress p(clamp_progress(static_cast<unsigned long long>(nc) * static_cast<unsigned long long>(nc2)), verbose);

#pragma omp parallel for num_threads(ncores) shared(d) schedule(dynamic)
  for (long long ii = 0; ii < nc_ll; ii++) {
    if (Progress::check_abort()) continue;
    const arma::uword i = static_cast<arma::uword>(ii);

    for (arma::uword j = 0; j < nc2; j++) {
      iter i_iter = m.begin_col(i);
      iter j_iter = m2.begin_col(j);
      double common = 0, i_count = 0, j_count = 0;

      while ((i_iter != m.end_col(i)) && (j_iter != m2.end_col(j))) {
        if (i_iter.row() == j_iter.row()) {
          i_count++; j_count++; common++; ++i_iter; ++j_iter;
        } else if (i_iter.row() < j_iter.row()) {
          i_count++; ++i_iter;
        } else {
          j_count++; ++j_iter;
        }
      }
      for (; i_iter != m.end_col(i);  ++i_iter) { i_count++; }
      for (; j_iter != m2.end_col(j); ++j_iter) { j_count++; }

      const double denom = i_count + j_count - common;
      const double sim   = (denom > 0.0) ? (common / denom) : 1.0;  // J(empty,empty)=1
      d(i, j) = dist ? (1.0 - sim) : sim;
    }
    p.increment(static_cast<unsigned long>(nc2));
  }

  stop_if_aborted(p);
  // Return type follows the data (see RESULT STORAGE in the header):
  // the distance form is non-zero for almost every column pair -> dense;
  // the similarity form is exactly 0 for disjoint pairs -> sparse.
  if (dist) return Rcpp::wrap(d);
  return Rcpp::wrap(arma::sp_mat(d));
}

// [[Rcpp::export]]
SEXP fastJacc(const arma::sp_mat& m, int ncores=1, bool verbose=true,
                      bool full=false, bool diag=true, bool dist=true)
{
  typedef arma::sp_mat::const_col_iterator iter;
  ncores = sanitize_ncores(ncores);
  const arma::uword ncol = m.n_cols;
  const long long   ncol_ll = static_cast<long long>(ncol);
  arma::mat d(ncol, ncol, arma::fill::zeros);

  Progress p(clamp_progress(tri_pairs(ncol, diag)), verbose);

#pragma omp parallel for num_threads(ncores) shared(d) schedule(dynamic)
  for (long long ii = 0; ii < ncol_ll; ii++) {
    if (Progress::check_abort()) continue;
    const arma::uword i = static_cast<arma::uword>(ii);
    unsigned long long steps = 0;

    for (arma::uword j = (diag ? i : i + 1); j < ncol; j++) {
      iter i_iter = m.begin_col(i);
      iter j_iter = m.begin_col(j);
      double common = 0, i_count = 0, j_count = 0;

      while ((i_iter != m.end_col(i)) && (j_iter != m.end_col(j))) {
        if (i_iter.row() == j_iter.row()) {
          i_count++; j_count++; common++; ++i_iter; ++j_iter;
        } else if (i_iter.row() < j_iter.row()) {
          i_count++; ++i_iter;
        } else {
          j_count++; ++j_iter;
        }
      }
      for (; i_iter != m.end_col(i); ++i_iter) { i_count++; }
      for (; j_iter != m.end_col(j); ++j_iter) { j_count++; }

      const double denom = i_count + j_count - common;
      const double sim   = (denom > 0.0) ? (common / denom) : 1.0;  // J(empty,empty)=1
      const double val   = dist ? (1.0 - sim) : sim;
      d(j, i) = val;
      if (full) d(i, j) = val;
      ++steps;
    }
    p.increment(static_cast<unsigned long>(steps));
  }

  stop_if_aborted(p);

  // The loop above only visits j == i when diag=true, so in SIMILARITY mode
  // (dist=false) the diagonal would otherwise be left at the zero-fill -- i.e.
  // a column would report 0 similarity with itself. The self-similarity is 1 by
  // definition, so write it unconditionally. (In DISTANCE mode the correct
  // self-distance is 0, which the zero-fill already gives, so nothing to do.)
  // Note this makes `diag` affect only how much work the loop does, not the
  // value on the diagonal, which is now always mathematically correct.
  if (!dist) d.diag().ones();

  // Return type follows the data (see RESULT STORAGE in the header):
  // the distance form is non-zero for almost every column pair -> dense;
  // the similarity form is exactly 0 for disjoint pairs -> sparse.
  if (dist) return Rcpp::wrap(d);
  return Rcpp::wrap(arma::sp_mat(d));
}

// ---------------------------------------------------------------------------
//  Manhattan (L1)
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
arma::mat fastManhattan(const arma::sp_mat& m, int ncores=1, bool verbose=true,
                           bool full=false, bool diag=true)
{
  typedef arma::sp_mat::const_col_iterator iter;
  ncores = sanitize_ncores(ncores);
  const arma::uword ncol = m.n_cols;
  const long long   ncol_ll = static_cast<long long>(ncol);
  arma::mat d(ncol, ncol, arma::fill::zeros);

  Progress p(clamp_progress(tri_pairs(ncol, diag)), verbose);

#pragma omp parallel for num_threads(ncores) shared(d) schedule(dynamic)
  for (long long ii = 0; ii < ncol_ll; ii++) {
    if (Progress::check_abort()) continue;
    const arma::uword i = static_cast<arma::uword>(ii);
    unsigned long long steps = 0;

    for (arma::uword j = (diag ? i : i + 1); j < ncol; j++) {
      iter i_iter = m.begin_col(i);
      iter j_iter = m.begin_col(j);
      double num = 0;

      while ((i_iter != m.end_col(i)) && (j_iter != m.end_col(j))) {
        if (i_iter.row() == j_iter.row()) {
          num += std::abs((*i_iter) - (*j_iter)); ++i_iter; ++j_iter;
        } else if (i_iter.row() < j_iter.row()) {
          num += std::abs(*i_iter); ++i_iter;
        } else {
          num += std::abs(*j_iter); ++j_iter;
        }
      }
      for (; i_iter != m.end_col(i); ++i_iter) { num += std::abs(*i_iter); }
      for (; j_iter != m.end_col(j); ++j_iter) { num += std::abs(*j_iter); }

      d(j, i) = num;
      if (full) d(i, j) = num;
      ++steps;
    }
    p.increment(static_cast<unsigned long>(steps));
  }

  stop_if_aborted(p);
  return d;
}

// [[Rcpp::export]]
arma::mat fastManhattan2(const arma::sp_mat& m, const arma::sp_mat& m2,
                            int ncores=1, bool verbose=true)
{
  if (m.n_rows != m2.n_rows) {
    Rcpp::stop("Mismatched row dimensions of m (" +
               std::to_string(static_cast<unsigned long long>(m.n_rows)) + ") and m2 (" +
               std::to_string(static_cast<unsigned long long>(m2.n_rows)) + ").");
  }
  typedef arma::sp_mat::const_col_iterator iter;
  ncores = sanitize_ncores(ncores);
  const arma::uword nc = m.n_cols, nc2 = m2.n_cols;
  const long long   nc_ll = static_cast<long long>(nc);
  arma::mat d(nc, nc2, arma::fill::zeros);

  Progress p(clamp_progress(static_cast<unsigned long long>(nc) * static_cast<unsigned long long>(nc2)), verbose);

#pragma omp parallel for num_threads(ncores) shared(d) schedule(dynamic)
  for (long long ii = 0; ii < nc_ll; ii++) {
    if (Progress::check_abort()) continue;
    const arma::uword i = static_cast<arma::uword>(ii);

    for (arma::uword j = 0; j < nc2; j++) {
      iter i_iter = m.begin_col(i);
      iter j_iter = m2.begin_col(j);
      double num = 0;

      while ((i_iter != m.end_col(i)) && (j_iter != m2.end_col(j))) {
        if (i_iter.row() == j_iter.row()) {
          num += std::abs((*i_iter) - (*j_iter)); ++i_iter; ++j_iter;
        } else if (i_iter.row() < j_iter.row()) {
          num += std::abs(*i_iter); ++i_iter;
        } else {
          num += std::abs(*j_iter); ++j_iter;
        }
      }
      for (; i_iter != m.end_col(i);  ++i_iter) { num += std::abs(*i_iter); }
      for (; j_iter != m2.end_col(j); ++j_iter) { num += std::abs(*j_iter); }

      d(i, j) = num;
    }
    p.increment(static_cast<unsigned long>(nc2));
  }

  stop_if_aborted(p);
  return d;
}

// ---------------------------------------------------------------------------
//  Euclidean (L2)
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
arma::mat fastEuclidean(const arma::sp_mat& m, int ncores=1, bool verbose=true,
                           bool full=false, bool diag=true)
{
  typedef arma::sp_mat::const_col_iterator iter;
  ncores = sanitize_ncores(ncores);
  const arma::uword ncol = m.n_cols;
  const long long   ncol_ll = static_cast<long long>(ncol);
  arma::mat d(ncol, ncol, arma::fill::zeros);

  Progress p(clamp_progress(tri_pairs(ncol, diag)), verbose);

#pragma omp parallel for num_threads(ncores) shared(d) schedule(dynamic)
  for (long long ii = 0; ii < ncol_ll; ii++) {
    if (Progress::check_abort()) continue;
    const arma::uword i = static_cast<arma::uword>(ii);
    unsigned long long steps = 0;

    for (arma::uword j = (diag ? i : i + 1); j < ncol; j++) {
      iter i_iter = m.begin_col(i);
      iter j_iter = m.begin_col(j);
      double num = 0;

      while ((i_iter != m.end_col(i)) && (j_iter != m.end_col(j))) {
        if (i_iter.row() == j_iter.row()) {
          const double diff = (*i_iter) - (*j_iter); num += diff * diff; ++i_iter; ++j_iter;
        } else if (i_iter.row() < j_iter.row()) {
          num += (*i_iter) * (*i_iter); ++i_iter;
        } else {
          num += (*j_iter) * (*j_iter); ++j_iter;
        }
      }
      for (; i_iter != m.end_col(i); ++i_iter) { num += (*i_iter) * (*i_iter); }
      for (; j_iter != m.end_col(j); ++j_iter) { num += (*j_iter) * (*j_iter); }

      const double val = std::sqrt(num);
      d(j, i) = val;
      if (full) d(i, j) = val;
      ++steps;
    }
    p.increment(static_cast<unsigned long>(steps));
  }

  stop_if_aborted(p);
  return d;
}

// [[Rcpp::export]]
arma::mat fastEuclidean2(const arma::sp_mat& m, const arma::sp_mat& m2,
                            int ncores=1, bool verbose=true)
{
  if (m.n_rows != m2.n_rows) {
    Rcpp::stop("Mismatched row dimensions of m (" +
               std::to_string(static_cast<unsigned long long>(m.n_rows)) + ") and m2 (" +
               std::to_string(static_cast<unsigned long long>(m2.n_rows)) + ").");
  }
  typedef arma::sp_mat::const_col_iterator iter;
  ncores = sanitize_ncores(ncores);
  const arma::uword nc = m.n_cols, nc2 = m2.n_cols;
  const long long   nc_ll = static_cast<long long>(nc);
  arma::mat d(nc, nc2, arma::fill::zeros);

  Progress p(clamp_progress(static_cast<unsigned long long>(nc) * static_cast<unsigned long long>(nc2)), verbose);

#pragma omp parallel for num_threads(ncores) shared(d) schedule(dynamic)
  for (long long ii = 0; ii < nc_ll; ii++) {
    if (Progress::check_abort()) continue;
    const arma::uword i = static_cast<arma::uword>(ii);

    for (arma::uword j = 0; j < nc2; j++) {
      iter i_iter = m.begin_col(i);
      iter j_iter = m2.begin_col(j);
      double num = 0;

      while ((i_iter != m.end_col(i)) && (j_iter != m2.end_col(j))) {
        if (i_iter.row() == j_iter.row()) {
          const double diff = (*i_iter) - (*j_iter); num += diff * diff; ++i_iter; ++j_iter;
        } else if (i_iter.row() < j_iter.row()) {
          num += (*i_iter) * (*i_iter); ++i_iter;
        } else {
          num += (*j_iter) * (*j_iter); ++j_iter;
        }
      }
      for (; i_iter != m.end_col(i);  ++i_iter) { num += (*i_iter) * (*i_iter); }
      for (; j_iter != m2.end_col(j); ++j_iter) { num += (*j_iter) * (*j_iter); }

      d(i, j) = std::sqrt(num);
    }
    p.increment(static_cast<unsigned long>(nc2));
  }

  stop_if_aborted(p);
  return d;
}

// ---------------------------------------------------------------------------
//  Jensen-Shannon distance  (dense input; columns assumed to be non-negative,
//  and should be normalised to sum to 1 for a true JS divergence)
// ---------------------------------------------------------------------------
//
//  Per-coordinate summand (0*log(0) := 0 applied to each x*log(x) term):
//      P*logP + Q*logQ - (P+Q)*log((P+Q)/2)
//    = xlogx(P) + xlogx(Q) - xlogx(P+Q) + (P+Q)*log(2)
//  The raw sum equals 2 * JSD, because the standard Jensen-Shannon divergence
//  carries a factor 1/2. The kernel therefore uses
//      v = 0.5 * sum(summand) = JSD
//  and returns sqrt(v) = sqrt(JSD), the STANDARD Jensen-Shannon distance (a true
//  metric, bounded by sqrt(log 2) in nats).
//
//  BEHAVIOUR CHANGE vs the original: the original returned sqrt(2 * JSD), i.e.
//  sqrt(2) ~ 1.4142 times larger. Multiply results by sqrt(2) to reproduce the
//  old values.

// [[Rcpp::export]]
arma::mat fastJS(const arma::mat& m, int ncores=1, bool verbose=true,
                    bool full=false, bool diag=true)
{
  if (m.n_elem > 0 && (!m.is_finite() || m.min() < 0.0)) {
    Rcpp::stop("fastJS requires finite, non-negative input "
               "(columns should be normalised to sum to 1 for a true JS divergence).");
  }
  ncores = sanitize_ncores(ncores);
  const arma::uword ncol = m.n_cols;
  const long long   ncol_ll = static_cast<long long>(ncol);
  arma::mat d(ncol, ncol, arma::fill::zeros);
  const double log2 = std::log(2.0);

  Progress p(clamp_progress(tri_pairs(ncol, diag)), verbose);

#pragma omp parallel for num_threads(ncores) shared(d) schedule(dynamic)
  for (long long ii = 0; ii < ncol_ll; ii++) {
    if (Progress::check_abort()) continue;
    const arma::uword i = static_cast<arma::uword>(ii);
    const arma::vec Pi   = m.col(i);
    const arma::vec xlPi = xlogx(Pi);
    unsigned long long steps = 0;

    for (arma::uword j = (diag ? i : i + 1); j < ncol; j++) {
      if (i == j) {
        // Self-comparison: JS(P, P) = 0 exactly. Computing it would instead
        // return sqrt of accumulated round-off -- and sqrt magnifies that
        // (sqrt has an infinite derivative at 0), so a divergence of ~1e-17
        // surfaces as a distance of ~3e-9. d is zero-filled, so skipping the
        // computation leaves a bit-exact 0.
        ++steps;
        continue;
      }
      const arma::vec Qj = m.col(j);
      const arma::vec s  = xlPi + xlogx(Qj) - xlogx(Pi + Qj) + (Pi + Qj) * log2;
      // 0.5 * sum(...) is the Jensen-Shannon DIVERGENCE; its sqrt is the
      // standard Jensen-Shannon DISTANCE.
      const double v = 0.5 * arma::accu(s);
      if (v > 0.0) {                       // d is zero-filled, so v<=0 -> a real 0; guards sqrt(neg)
        const double val = std::sqrt(v);
        d(j, i) = val;
        if (full) d(i, j) = val;
      }
      ++steps;
    }
    p.increment(static_cast<unsigned long>(steps));
  }

  stop_if_aborted(p);
  return d;
}

// [[Rcpp::export]]
arma::mat fastJS2(const arma::mat& m, const arma::mat& m2,
                     int ncores=1, bool verbose=true)
{
  if ((m.n_elem  > 0 && (!m.is_finite()  || m.min()  < 0.0)) ||
      (m2.n_elem > 0 && (!m2.is_finite() || m2.min() < 0.0))) {
    Rcpp::stop("fastJS2 requires finite, non-negative input "
               "(columns should be normalised to sum to 1 for a true JS divergence).");
  }
  if (m.n_rows != m2.n_rows) {
    Rcpp::stop("Mismatched row dimensions of m (" +
               std::to_string(static_cast<unsigned long long>(m.n_rows)) + ") and m2 (" +
               std::to_string(static_cast<unsigned long long>(m2.n_rows)) + ").");
  }
  ncores = sanitize_ncores(ncores);
  const arma::uword nc = m.n_cols, nc2 = m2.n_cols;
  const long long   nc_ll = static_cast<long long>(nc);
  arma::mat d(nc, nc2, arma::fill::zeros);
  const double log2 = std::log(2.0);

  Progress p(clamp_progress(static_cast<unsigned long long>(nc) * static_cast<unsigned long long>(nc2)), verbose);

#pragma omp parallel for num_threads(ncores) shared(d) schedule(dynamic)
  for (long long ii = 0; ii < nc_ll; ii++) {
    if (Progress::check_abort()) continue;
    const arma::uword i = static_cast<arma::uword>(ii);
    const arma::vec Pi   = m.col(i);
    const arma::vec xlPi = xlogx(Pi);

    for (arma::uword j = 0; j < nc2; j++) {
      const arma::vec Qj = m2.col(j);
      const arma::vec s  = xlPi + xlogx(Qj) - xlogx(Pi + Qj) + (Pi + Qj) * log2;
      // 0.5 * sum(...) is the Jensen-Shannon DIVERGENCE; its sqrt is the
      // standard Jensen-Shannon DISTANCE.
      const double v = 0.5 * arma::accu(s);
      if (v > 0.0) d(i, j) = std::sqrt(v);
    }
    p.increment(static_cast<unsigned long>(nc2));
  }

  stop_if_aborted(p);
  return d;
}
