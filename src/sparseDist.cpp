// =============================================================================
//  sparseDist.cpp  --  patched version
// =============================================================================
//
//  Column-wise distance / similarity kernels over Armadillo sparse (and, for
//  Jensen-Shannon, dense) matrices, parallelised with OpenMP and reporting
//  progress via RcppProgress.
//
//  ---------------------------------------------------------------------------
//  WHAT CHANGED RELATIVE TO THE ORIGINAL
//  ---------------------------------------------------------------------------
//  Correctness fixes
//    * The result buffer `d` is now ALWAYS zero-initialised. The original left
//      it uninitialised in fastJacc/fastManhattan/fastEuclidean/fastJS, so the
//      un-written (upper) triangle could be converted to sp_mat as garbage on
//      Armadillo builds that do not zero-fill mat(n,n).
//    * fastJS / fastJS2: the divergence is now computed with the 0*log(0)=0
//      convention applied to each x*log(x) term individually. The original used
//      find_finite() over the whole per-coordinate summand, which turned a
//      coordinate into NaN (and dropped it) whenever exactly one distribution
//      was zero there -- silently discarding the legitimate x*log(2)
//      contribution and under-counting the divergence for distributions with
//      differing support. Also guards sqrt() against tiny negative round-off
//      (was `if (v != 0) sqrt(v)`, which could yield NaN).
//    * Degenerate columns no longer poison the output with NaN:
//        - correlation of a zero-variance / empty column (undefined, 0/0) is
//          treated as 0 correlation;
//        - Jaccard of two empty columns (0/0) is defined as similarity 1
//          (distance 0), i.e. two empty sets are identical.
//    * fastJacc: removed the `d(j,i) = d(j,i) = ...` double-assignment typo.
//    * Progress totals are computed exactly in 64-bit arithmetic, fixing the old
//      int/unsigned-int total that overflowed past ~46k/65k columns (and the
//      wrong diag=false count). RcppProgress itself stores counts as `unsigned
//      long` (32-bit on LLP64 platforms such as Windows), so clamp_progress()
//      saturates the total at ULONG_MAX to keep the progress *display* from
//      wrapping; that bound is unreachable given the dense O(n_cols^2) result.
//    * Loop counters widened to 64-bit signed, avoiding overflow for very large
//      matrices and remaining portable across OpenMP versions.
//    * A user interrupt now raises an R error ("Computation interrupted by the
//      user.") instead of silently returning a partially computed matrix.
//    * fastCorr now forces an exact zero diagonal (distance) / unit diagonal
//      (similarity) via an i==j special case, so degenerate columns no longer
//      leave a non-zero self-distance on the diagonal.
//    * fastJS: a self-comparison (i == j) short-circuits to an exact 0 instead
//      of sqrt(accumulated round-off). Only the single-matrix form can do this;
//      fastJS2 cannot tell that a column of m and one of m2 are the same data.
//    * fastCorr / fastCorr2: the correlation is clamped to [-1, 1] before the
//      distance is formed. arma::cor can land a few ULP outside that range for
//      near-identical columns, which made 1 - c a tiny NEGATIVE distance --
//      enough to upset as.dist(), sqrt() or an MDS downstream.
//    * fastCorr / fastJacc: the diagonal is correct even when diag=false. The
//      loop only visits j==i when diag=true, so in SIMILARITY mode (dist=false)
//      the diagonal was previously left at 0 -- a column reporting 0 similarity
//      with itself. It is now set to 1 unconditionally after the parallel
//      region. (Distance mode was already right: self-distance 0 = zero-fill.)
//    * fastJS / fastJS2 now return the STANDARD Jensen-Shannon distance
//      sqrt(JSD); the original returned sqrt(2*JSD), a factor of sqrt(2) larger.
//    * fastJS / fastJS2 validate that inputs are finite and non-negative, so
//      invalid (e.g. negative) inputs fail loudly instead of silently producing
//      a meaningless-but-finite result. (Column normalisation remains the
//      caller's responsibility; it is not enforced.)
//    * Dimension-mismatch error messages use std::to_string instead of an int
//      cast, so they are correct for dimensions above INT_MAX.
//
//  New in this version
//    * fastCosine / fastCosine2: cosine similarity and 1 - cosine distance.
//      The dot product is a merge walk over the overlapping support and the
//      column norms are precomputed, so no column is ever densified -- unlike
//      the correlation and covariance kernels.
//
//  Consistency / API changes  (NOTE: may require updating R-side wrappers)
//    * `full` now has the SAME meaning in every function: full=false returns the
//      lower triangle only (plus the diagonal iff diag=true); full=true fills
//      both triangles (symmetric). Previously fastCorr/fastCov ignored `full`
//      and were ALWAYS symmetric. If you relied on that, pass full=TRUE (or
//      symmetrise on the R side, as the other metrics already required).
//    * fastCov lost its `dist` argument: it was accepted but never used
//      (there is no canonical "distance" form of a covariance).
//    * RETURN TYPES CHANGED: distance-valued kernels now return a DENSE matrix
//      and only coefficient-valued ones return sp_mat (details under RESULT
//      STORAGE below). R code that assumed a dgCMatrix from every function --
//      e.g. calling Matrix-only methods on the result -- must be updated.
//
//  Performance
//    * The two-matrix kernels no longer use `collapse(2)`; the i-th column is
//      densified once per i instead of once per (i,j) pair. Consequence: the
//      first matrix is the parallelised dimension, so for best scaling pass the
//      matrix with more columns as `m`.
//    * schedule(dynamic) balances the triangular loops (later i do less work).
//    * Progress/abort are checked once per outer column instead of once per
//      pair, cutting per-pair R_ToplevelExec overhead on the master thread.
//      (RcppProgress is already thread-safe: non-master threads short-circuit
//      before touching R, and increment() is internally atomic.)
//
//  RESULT STORAGE (why the accumulator is dense; what each kernel returns)
//    * The accumulator `d` is a DENSE arma::mat in every kernel, and that is
//      required for CORRECTNESS, not merely for speed. arma::sp_mat is CSC:
//      inserting a new element shifts the packed value/index arrays after it
//      (O(nnz) per insert) and may reallocate, so concurrent inserts from
//      OpenMP threads would corrupt the structure. A dense buffer gives every
//      (i,j) a fixed address, which is exactly what makes the lock-free
//      parallel fill valid. Accumulate dense, convert once at the end.
//    * Return types follow the data rather than being uniformly sparse:
//        - DISTANCE-valued results are dense (arma::mat). With sparse input
//          most column pairs share nothing, which makes their DISTANCE
//          non-zero (1 - 0 for Jaccard/correlation distance, or a positive
//          norm), so a distance matrix is nearly 100% full. Holding that as
//          sp_mat costs ~12-16 bytes per stored entry against 8 dense, and the
//          conversion holds the dense buffer plus the sparse copy at once
//          (~3x peak). Dense also removes the true-zero/structural-zero
//          ambiguity -- 0 now means "identical columns" -- and a lower
//          triangle feeds as.dist() directly.
//        - COEFFICIENT/similarity results stay sparse (arma::sp_mat), where the
//          zeros are real: Jaccard similarity is exactly 0 for disjoint column
//          pairs, the common case in sparse data.
//      Concretely: fastEuclidean(2), fastManhattan(2), fastJS(2) -> dense;
//      fastCov(2) -> sparse; fastJacc(2), fastCosine(2), fastCorr(2) -> dense
//      when dist=true,
//      sparse when dist=false (hence declared SEXP: the R-level type depends
//      on that flag).
//      CAVEAT: Jaccard and cosine similarities are sparse because of the DATA
//      -- a pair with disjoint support has exactly zero similarity. For
//      correlation and covariance, columns with disjoint support still give a
//      small non-zero value, so their sparsity is only structural: the
//      unwritten upper triangle (full=false) plus degenerate columns mapped to
//      0. With full=true and no degenerate columns those results are
//      effectively dense, and sp_mat mostly adds index overhead.
//
//  KNOWN LIMITATION
//    * The cosine kernels additionally hold a unit-normalised copy of the input
//      in plain CSC arrays, roughly the size of the input matrix (8 bytes per
//      stored value plus 8 per row index). That is the price of normalising
//      once instead of per pair; it is O(nnz) on top of the dense accumulator
//      below, and is released when the call returns.
//    * Peak memory is O(n_cols^2): the all-pairs result is materialised in full
//      before being returned. That is inherent to returning the whole matrix;
//      only a blocked or streaming API (emitting pairs above a threshold) would
//      avoid it.
// =============================================================================

#include <RcppArmadillo.h>
#include <progress.hpp>
#include <cmath>
#include <string>
#include <vector>
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


// L2 norm of every column, computed once. Cosine needs ||x|| for each column
// in every pair, so recomputing it inside the loop would cost an extra factor
// of n_cols. One pass over the stored non-zeros is negligible beside the
// O(n_cols^2) pair loop.
// Unit-normalised copy of a sparse matrix, held in plain CSC arrays.
//
// Normalising ONCE and taking a plain dot product of unit columns is what makes
// the cosine numerically safe. Computing <x,y> / (||x||*||y||) directly is not:
// for columns holding ~1e200 the dot product and the product of norms both
// overflow to Inf and the cosine becomes Inf/Inf = NaN (and a clamp cannot fix
// a NaN, since every comparison against it is false); for ~1e-200 both
// underflow to 0 and the cosine is reported as 0 when it is really 1. After
// normalisation every stored value is at most 1 in magnitude and the dot
// product is bounded by 1 through Cauchy-Schwarz, so neither can happen.
//
// It is also faster: the pair loop does no division at all, whereas dividing
// by ||x||*||y|| costs one per pair, and scaling inside the loop would cost two
// per overlapping entry.
//
// Per-column norms use the scaled sum-of-squares recurrence (as in BLAS dnrm2),
// so a norm is itself computed without overflowing or underflowing. Cost is one
// pass over the stored non-zeros; memory is about the size of the input matrix.
struct UnitCsc {
  std::vector<arma::uword> rows;    // nnz row indices, ascending within a column
  std::vector<double>      vals;    // nnz values, each column scaled to unit norm
  std::vector<std::size_t> cptr;    // n_cols + 1 column offsets
  // Largest |value| in each column: exactly 0 when the column holds no
  // non-zero entry, and always finite because the input is validated. This is
  // all the pair loop needs -- storing the true L2 norm instead would be less
  // precise, since a norm can itself overflow to Inf for extreme input.
  std::vector<double>      scale;
};

static inline UnitCsc unit_csc(const arma::sp_mat& m) {
  typedef arma::sp_mat::const_col_iterator iter;
  UnitCsc U;
  const arma::uword nc = m.n_cols;
  U.cptr.assign(static_cast<std::size_t>(nc) + 1u, 0u);
  U.scale.assign(static_cast<std::size_t>(nc), 0.0);
  U.rows.reserve(m.n_nonzero);
  U.vals.reserve(m.n_nonzero);

  for (arma::uword j = 0; j < nc; ++j) {
    double scale = 0.0, ssq = 1.0;
    for (iter it = m.begin_col(j); it != m.end_col(j); ++it) {
      const double v = *it;
      // Reject non-finite input HERE, in the serial pass before any parallel
      // region, where raising an R error is safe. Without this a stored NaN
      // makes the column look empty (NaN never compares greater, so `scale`
      // stays 0) and a stored Inf normalises to Inf/Inf = NaN, which the guard
      // in unit_cosine() would turn into a plausible-looking 0. Silently
      // returning a believable number for invalid input is worse than either
      // erroring or propagating NaN.
      if (!std::isfinite(v)) {
        Rcpp::stop("Cosine requires finite input (found NA, NaN or Inf).");
      }
      if (v == 0.0) continue;
      const double a = std::fabs(v);
      if (scale < a) { const double r = scale / a; ssq = 1.0 + ssq * r * r; scale = a; }
      else           { const double r = a / scale; ssq += r * r; }
    }
    const double sq = std::sqrt(ssq);
    U.scale[static_cast<std::size_t>(j)] = scale;

    for (iter it = m.begin_col(j); it != m.end_col(j); ++it) {
      U.rows.push_back(it.row());
      U.vals.push_back((scale == 0.0) ? 0.0 : ((*it) / scale) / sq);
    }
    U.cptr[static_cast<std::size_t>(j) + 1u] = U.rows.size();
  }
  return U;
}

// Dot product of two unit columns: a merge walk over the overlapping support.
static inline double unit_dot(const UnitCsc& A, arma::uword i,
                              const UnitCsc& B, arma::uword j) {
  std::size_t a  = A.cptr[static_cast<std::size_t>(i)];
  const std::size_t ae = A.cptr[static_cast<std::size_t>(i) + 1u];
  std::size_t b  = B.cptr[static_cast<std::size_t>(j)];
  const std::size_t be = B.cptr[static_cast<std::size_t>(j) + 1u];
  double s = 0.0;
  while (a < ae && b < be) {
    if (A.rows[a] == B.rows[b]) { s += A.vals[a] * B.vals[b]; ++a; ++b; }
    else if (A.rows[a] < B.rows[b]) { ++a; }
    else { ++b; }
  }
  return s;
}

// Cosine of two unit columns.
//
// EMPTY COLUMNS. The zero vector has no direction, so the cosine is undefined
// whenever either column is empty and the values below are conventions:
//   * both empty  -> similarity 1 (distance 0). Two all-zero columns are the
//     same vector, so treating them as identical keeps d(x, x) = 0, and it
//     matches the Jaccard kernel, where two empty sets also have similarity 1.
//     It further means sparseDist(X, X) keeps a zero diagonal for empty
//     columns, which the two-matrix kernel could not otherwise achieve: it is
//     given two distinct objects and cannot know a column is being compared
//     with itself.
//   * exactly one empty -> similarity 0 (distance 1), as an empty set has
//     Jaccard similarity 0 with a non-empty one.
//
// The NaN branch is unreachable: unit_csc() rejects non-finite input and every
// normalised value lies in [-1, 1], so the dot product is bounded by 1 through
// Cauchy-Schwarz. It is kept only because a clamp alone would silently pass a
// NaN through, which is exactly how an earlier version of this kernel turned
// overflow into a plausible-looking answer.
static inline double unit_cosine(const UnitCsc& A, arma::uword i,
                                 const UnitCsc& B, arma::uword j) {
  const bool ea = (A.scale[static_cast<std::size_t>(i)] == 0.0);
  const bool eb = (B.scale[static_cast<std::size_t>(j)] == 0.0);
  if (ea || eb) return (ea && eb) ? 1.0 : 0.0;
  const double c = unit_dot(A, i, B, j);
  if (!(c == c)) return 0.0;                 // unreachable; see above
  if (c >  1.0)  return  1.0;                // round-off only
  if (c < -1.0)  return -1.0;
  return c;
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
//  Cosine
// ---------------------------------------------------------------------------
//
//  cos(x, y) = <x, y> / (||x|| * ||y||)
//
//  The dot product only receives contributions where the two supports overlap,
//  so a merge walk over the stored non-zeros is enough and neither column is
//  ever densified. Column norms are precomputed once.
//
//  Input must be finite: NA, NaN and Inf are rejected with an error rather
//  than folded into a plausible-looking result.
//
//  ZERO-NORM (empty) COLUMNS. The cosine is undefined, 0/0. Two cases:
//    * a known SELF-comparison -- fastCosine with i == j -- is fixed BY
//      CONVENTION at similarity 1 / distance 0, including for an all-zero
//      column, where the cosine is genuinely undefined;
//    * TWO empty columns are also similarity 1 / distance 0: they are the same
//      vector, and this matches the Jaccard convention that two empty sets are
//      identical. It is what lets sparseDist(X, X) keep a zero diagonal for
//      empty columns even though fastCosine2 cannot know a column is being
//      compared with itself;
//    * exactly ONE empty column gives similarity 0 / distance 1, as an empty
//      set has Jaccard similarity 0 with a non-empty one.
//  For NON-empty columns the two-matrix form still differs from the
//  single-matrix one by round-off on the diagonal (~1e-16), because it actually
//  evaluates the dot product instead of short-circuiting. sparseKNN() restores
//  the exact self-values after each block, so the blocked search is unaffected.
//
//  With dist=true the DISTANCE 1 - cos is returned, the usual convention for
//  comparing embeddings. Note it is not a metric: the triangle inequality can
//  fail, so use the angular distance acos(cos)/pi if a true metric is needed --
//  itself a metric only among columns of non-zero norm, since the zero-vector
//  values above are conventional rather than derived.
//  For non-negative data (counts, TF-IDF) it lies in [0, 1]; for signed data
//  (most learned embeddings) in [0, 2].

// [[Rcpp::export]]
SEXP fastCosine(const arma::sp_mat& m, int ncores=1, bool verbose=true,
                bool full=false, bool diag=true, bool dist=true)
{
  ncores = sanitize_ncores(ncores);
  const arma::uword ncol = m.n_cols;
  const long long   ncol_ll = static_cast<long long>(ncol);
  arma::mat d(ncol, ncol, arma::fill::zeros);
  const UnitCsc U = unit_csc(m);

  Progress p(clamp_progress(tri_pairs(ncol, diag)), verbose);

#pragma omp parallel for num_threads(ncores) shared(d) schedule(dynamic)
  for (long long ii = 0; ii < ncol_ll; ii++) {
    if (Progress::check_abort()) continue;
    const arma::uword i = static_cast<arma::uword>(ii);
    unsigned long long steps = 0;

    for (arma::uword j = (diag ? i : i + 1); j < ncol; j++) {
      double val;
      if (i == j) {
        // Self-comparison is fixed by convention at similarity 1 / distance 0.
        // For a non-empty column that is also the exact value; for an all-zero
        // column the cosine is undefined and this is the package convention.
        val = dist ? 0.0 : 1.0;
      } else {
        const double c = unit_cosine(U, i, U, j);
        val = dist ? (1.0 - c) : c;
      }
      d(j, i) = val;
      if (full) d(i, j) = val;
      ++steps;
    }
    p.increment(static_cast<unsigned long>(steps));
  }

  stop_if_aborted(p);

  // As in fastCorr/fastJacc: the loop reaches j == i only when diag=true, so in
  // SIMILARITY mode the diagonal would otherwise stay at the zero-fill.
  if (!dist) d.diag().ones();

  // Distances are dense (nearly every pair is non-zero); similarities are
  // sparse, and genuinely so -- disjoint columns give a dot product of exactly
  // zero. See RESULT STORAGE in the header.
  if (dist) return Rcpp::wrap(d);
  return Rcpp::wrap(arma::sp_mat(d));
}

// [[Rcpp::export]]
SEXP fastCosine2(const arma::sp_mat& m, const arma::sp_mat& m2,
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
  const UnitCsc U  = unit_csc(m);
  const UnitCsc U2 = unit_csc(m2);

  Progress p(clamp_progress(static_cast<unsigned long long>(nc) *
                            static_cast<unsigned long long>(nc2)), verbose);

#pragma omp parallel for num_threads(ncores) shared(d) schedule(dynamic)
  for (long long ii = 0; ii < nc_ll; ii++) {
    if (Progress::check_abort()) continue;
    const arma::uword i = static_cast<arma::uword>(ii);

    for (arma::uword j = 0; j < nc2; j++) {
      const double c = unit_cosine(U, i, U2, j);
      d(i, j) = dist ? (1.0 - c) : c;
    }
    p.increment(static_cast<unsigned long>(nc2));
  }

  stop_if_aborted(p);
  if (dist) return Rcpp::wrap(d);
  return Rcpp::wrap(arma::sp_mat(d));
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
