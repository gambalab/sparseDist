// =============================================================================
//  topKRows.cpp -- running k-best-per-ROW reduction of a dense distance block.
//
//  topKBlock() reduces a slab column-wise: each query column of D is scanned
//  once and its k best reference rows are final by the time the block is done.
//  The reverse direction is not final: row i of D holds only the comparisons
//  against ONE block of queries, so its k best must be accumulated as blocks
//  stream past. That running state is what this kernel maintains.
//
//  Two things make the naive version slow, and both are addressed here.
//
//  MEMORY ORDER. D is column-major, so walking a row strides by n_rows * 8
//  bytes and touches a fresh cache line per element -- for a 156k x 2048 slab
//  that is a cache miss per candidate, which would cost more than the distance
//  computation it is meant to save. Instead the rows are processed in tiles:
//  for a tile of `tile_rows` consecutive rows the kernel walks D column by
//  column, reading `tile_rows` CONTIGUOUS doubles from each, and updates
//  tile_rows heaps in step. At the default 64 rows a column slice is 512 bytes
//  and the tile's heaps are ~50 KB, so the working set stays in L2.
//
//  STATE SIZE. The running state is n_rows x k, not n_rows x k per block:
//  accumulating per-block partials and merging at the end would cost
//  n_blocks times more memory (7 GB rather than 90 MB at 156k rows, k = 50).
//
//  TIE-BREAKING. Candidates are compared on (value, GLOBAL query index) with
//  the lower index winning, exactly as in topKBlock. Because blocks arrive in
//  increasing column order and a candidate replaces an incumbent only when
//  STRICTLY better, an earlier query always beats a later one at equal
//  distance. The result is therefore independent of block_size and of the
//  thread count -- which is what makes it comparable to sparseKNN(Y, X).
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

typedef std::pair<double, arma::uword> Cand;   // (value, 0-based GLOBAL col idx)

// Strict weak ordering: "a is a better neighbour than b". Heap top is the
// WORST kept candidate. Identical in form to topK.cpp's comparator, so the two
// directions rank ties the same way.
struct BetterRow {
  bool decreasing;
  explicit BetterRow(bool d) : decreasing(d) {}
  bool operator()(const Cand& a, const Cand& b) const {
    if (a.first != b.first) return decreasing ? (a.first > b.first) : (a.first < b.first);
    return a.second < b.second;
  }
};

}  // namespace

// Fold one distance block into a running per-row top-k.
//
//   D          dense block; D(i, b) relates reference column i to the query
//              column col_offset + b (0-based).
//   k          neighbours to keep per row.
//   decreasing keep the k LARGEST values (similarity) rather than the smallest.
//   best_idx   n_rows x k, 1-based GLOBAL query indices, NA for empty slots.
//   best_val   n_rows x k, matching values, NA for empty slots.
//              BOTH ARE MUTATED IN PLACE -- see the warning below.
//   col_offset 0-based index of D's first column in the full query set.
//   self_col   for each ROW of D, the 1-based GLOBAL query index that is the
//              row's own self-comparison, or 0 when there is none.
//   ncores     threads.
//   tile_rows  rows per tile; tune only if profiling says so.
//
// WARNING: best_idx and best_val are written through their underlying R
// storage without duplication. This is deliberate -- copying an n_rows x k
// state on every block would dominate the kernel -- but it means the caller
// must own these matrices exclusively and must never let R's copy-on-modify
// semantics be relied upon for them. sparseKNNCross() allocates them fresh and
// never aliases them; no other caller should exist.
//
// A MAYBE_SHARED() guard was measured and rejected, not overlooked. Under
// R 4.6.1 / Rcpp 1.1.2 it returns TRUE for EVERY argument reaching a native
// helper -- including an anonymous temporary with no other binding anywhere,
// and including the raw-SEXP path that bypasses the generated wrapper -- and
// equally TRUE for genuinely aliased objects. It therefore has no power to
// distinguish the hazard from ordinary argument passing, and a guard built on
// it would reject every valid call. MAYBE_REFERENCED() is strictly weaker and
// no better. The invariant is consequently held by construction in
// sparseKNNCross() (fresh allocation, never aliased, this helper not exported)
// and is structurally enforceable only by moving the state behind an XPtr --
// which buys nothing while there is a single caller, and would be the right
// answer only if the accumulation ever needs to be checkpointed, resumed, or
// handed back to the caller between blocks.
//
// Not exported to users; called from sparseKNNCross().
// [[Rcpp::export]]
void topKRowsAccum(const arma::mat& D, int k, bool decreasing,
                   Rcpp::IntegerMatrix best_idx, Rcpp::NumericMatrix best_val,
                   int col_offset, const Rcpp::IntegerVector& self_col,
                   int ncores = 1, int tile_rows = 64) {
  const arma::uword nrow = D.n_rows;
  const arma::uword ncol = D.n_cols;

  if (k < 1) Rcpp::stop("k must be >= 1.");
  if (ncores < 1) ncores = 1;
  if (tile_rows < 1) tile_rows = 1;
  if (col_offset < 0) Rcpp::stop("col_offset must be >= 0.");

  const arma::uword K = static_cast<arma::uword>(k);

  if (static_cast<arma::uword>(best_idx.nrow()) != nrow ||
      static_cast<arma::uword>(best_idx.ncol()) != K ||
      static_cast<arma::uword>(best_val.nrow()) != nrow ||
      static_cast<arma::uword>(best_val.ncol()) != K) {
    Rcpp::stop("best_idx and best_val must both be nrow(D) x k.");
  }
  if (static_cast<arma::uword>(self_col.size()) != nrow) {
    Rcpp::stop("self_col must have one entry per row of D.");
  }

  // Copy self_col out of the R object before the parallel region: R/Rcpp
  // objects must not be read from worker threads through their API.
  std::vector<int> self(static_cast<std::size_t>(nrow));
  for (arma::uword i = 0; i < nrow; ++i) {
    const int sc = self_col[i];
    if (sc == NA_INTEGER || sc < 0) {
      Rcpp::stop("self_col entries must be 0 or a positive 1-based query index.");
    }
    self[static_cast<std::size_t>(i)] = sc;
  }

  // Raw pointers into the R matrices. Column-major: element (i, t) lives at
  // t * nrow + i. Only plain loads and stores happen through these inside the
  // parallel region -- no allocation, no R API.
  int*    const bi_p = best_idx.begin();
  double* const bv_p = best_val.begin();

  const BetterRow cmp(decreasing);
  const arma::uword TR = static_cast<arma::uword>(tile_rows);
  const long long n_tiles = static_cast<long long>((nrow + TR - 1) / TR);

  // One contiguous slab of heap storage per thread, allocated ONCE here rather
  // than a std::vector per row per tile. The vector-of-vectors version issued
  // one reserve() for every row of every block -- at 156k rows and 77 blocks
  // that is ~12 million small allocations for a single bidirectional search,
  // which becomes conspicuous precisely because the distance kernel is now
  // fast. Row r of the tile owned by thread t gets the fixed segment
  // [t*TR + r] * (K+1), and the standard heap algorithms operate on the
  // pointer range directly. The K+1 stride matches the reserve(K+1) of the
  // original so a push into a full heap never runs off its segment.
  // Never ask for more workers, or allocate more scratch, than there are tiles
  // to hand out. At the production geometry (thousands of tiles) this is a
  // no-op; it matters for small calls, where a 10-row problem with ncores = 64
  // would otherwise reserve 64 segments and open a 64-thread team to run a
  // single tile. team_threads is also what indexes heap_buf, so the two must
  // be derived from the same value.
  const int team_threads = std::max<int>(1, std::min<int>(ncores, static_cast<int>(n_tiles)));
  const std::size_t nthreads = static_cast<std::size_t>(team_threads);
  const std::size_t seg      = static_cast<std::size_t>(K) + 1;
  std::vector<Cand>       heap_buf(nthreads * static_cast<std::size_t>(TR) * seg);
  std::vector<arma::uword> heap_len(nthreads * static_cast<std::size_t>(TR), 0);

#pragma omp parallel for num_threads(team_threads) schedule(dynamic)
  for (long long tt = 0; tt < n_tiles; tt++) {
#ifdef _OPENMP
    const std::size_t tid = static_cast<std::size_t>(omp_get_thread_num());
#else
    const std::size_t tid = 0;
#endif
    Cand* const       hbase = heap_buf.data() + tid * static_cast<std::size_t>(TR) * seg;
    arma::uword* const hlen = heap_len.data() + tid * static_cast<std::size_t>(TR);

    const arma::uword r0 = static_cast<arma::uword>(tt) * TR;
    const arma::uword r1 = std::min(r0 + TR, nrow);
    const arma::uword nr = r1 - r0;

    // Seed each row's heap from the state carried in from previous blocks.
    // Slots are stored best-first, so filling in order and heapifying once is
    // correct and cheaper than K push_heap calls.
    for (arma::uword r = 0; r < nr; ++r) {
      Cand* const h = hbase + static_cast<std::size_t>(r) * seg;
      arma::uword n = 0;
      const arma::uword i = r0 + r;
      for (arma::uword t = 0; t < K; ++t) {
        const std::size_t at = static_cast<std::size_t>(t) * nrow + i;
        const int gi = bi_p[at];
        if (gi == NA_INTEGER || gi <= 0) break;      // slots fill left to right
        h[n++] = Cand(bv_p[at], static_cast<arma::uword>(gi - 1));
      }
      hlen[r] = n;
      std::make_heap(h, h + n, cmp);
    }

    // Column-major walk: for each query column, read the tile's rows as one
    // contiguous run of doubles.
    for (arma::uword b = 0; b < ncol; ++b) {
      const arma::uword g  = static_cast<arma::uword>(col_offset) + b;
      const int         g1 = static_cast<int>(g) + 1;         // 1-based
      const double* const col = D.colptr(b);

      for (arma::uword r = 0; r < nr; ++r) {
        const arma::uword i = r0 + r;
        if (self[static_cast<std::size_t>(i)] == g1) continue;  // own column

        const double v = col[i];
        if (!std::isfinite(v)) continue;      // NA/NaN never occupies a slot

        Cand* const  h  = hbase + static_cast<std::size_t>(r) * seg;
        arma::uword& hs = hlen[r];
        if (hs < K) {
          h[hs++] = Cand(v, g);
          std::push_heap(h, h + hs, cmp);
        } else if (cmp(Cand(v, g), h[0])) {
          // Strictly better only: at equal value the incumbent has the smaller
          // global index (blocks arrive in increasing order), so it must win.
          std::pop_heap(h, h + K, cmp);
          h[K - 1] = Cand(v, g);
          std::push_heap(h, h + K, cmp);
        }
      }
    }

    // Write the tile's state back, best first, padding unused slots.
    for (arma::uword r = 0; r < nr; ++r) {
      Cand* const h = hbase + static_cast<std::size_t>(r) * seg;
      const arma::uword n = hlen[r];
      std::sort_heap(h, h + n, cmp);
      const arma::uword i = r0 + r;
      for (arma::uword t = 0; t < K; ++t) {
        const std::size_t at = static_cast<std::size_t>(t) * nrow + i;
        if (t < n) {
          bv_p[at] = h[t].first;
          bi_p[at] = static_cast<int>(h[t].second) + 1;
        } else {
          bv_p[at] = NA_REAL;
          bi_p[at] = NA_INTEGER;
        }
      }
    }
  }
}
