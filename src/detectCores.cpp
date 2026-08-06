// Hardware thread count, plus a probe reporting whether this shared object was
// actually built with OpenMP support.
//
// Approach for the core count borrowed from RcppThread:
//   https://github.com/tnagler/RcppThread/commit/c26fc2b0d56555fa434c33352747822691334fe8
// -----------------------------------------------------------------------------

#include <Rcpp.h>

#include <thread>
#include <climits>

#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::plugins(openmp)]]

// Number of concurrent threads supported by the hardware.
// Returns at least 1: hardware_concurrency() yields 0 when the value is not
// computable, and a 0 thread count is never a useful answer for a caller.
//
// CAVEAT (see ompInfoCpp below): this reports the HARDWARE, and ignores any
// cgroup / cpuset restriction imposed by a batch scheduler. On a cluster node
// where the job has been allocated a subset of the CPUs, it therefore
// over-reports, and a caller sizing a thread pool from it will oversubscribe.
//
// [[Rcpp::export]]
int detectCoresCpp() {
  const unsigned int n = std::thread::hardware_concurrency();
  if (n == 0u) return 1;
  if (n > static_cast<unsigned int>(INT_MAX)) return INT_MAX;
  return static_cast<int>(n);
}

// Whether OpenMP is available in THIS build, and what the runtime thinks it
// has to work with.
//
// This exists because the absence of OpenMP is otherwise invisible: every
// kernel compiles and runs correctly without it, silently serial, and `ncores`
// simply has no effect. A default macOS build (Apple clang ships no OpenMP) is
// the common case. Benchmarks in particular must be able to refuse to record a
// timing from a build where the thread count is inert, rather than relying on
// the operator to remember.
//
// Returned fields:
//   openmp       TRUE when compiled with OpenMP.
//   spec         the _OPENMP macro, a yyyymm version stamp (e.g. 201511 for
//                OpenMP 4.5); NA when unsupported.
//   max_threads  omp_get_max_threads(): the size of the team a parallel region
//                would get right now, i.e. what OMP_NUM_THREADS and any earlier
//                omp_set_num_threads() have settled on.
//   num_procs    omp_get_num_procs(): processors available to the OpenMP
//                runtime. Under libgomp this RESPECTS the process affinity
//                mask, so on a scheduler-allocated node it reflects the actual
//                allocation.
//   hw_threads   detectCoresCpp(): the raw hardware count, which does NOT.
//
// num_procs < hw_threads therefore means the process is confined to a subset of
// the machine -- exactly the situation in which sizing a thread pool from
// hw_threads oversubscribes. Reporting both makes that visible instead of
// leaving it to be inferred from a bad timing.
//
// Not exported to users; called for provenance and capability checks.
// [[Rcpp::export]]
Rcpp::List ompInfoCpp() {
#ifdef _OPENMP
  return Rcpp::List::create(
      Rcpp::Named("openmp")      = true,
      Rcpp::Named("spec")        = static_cast<int>(_OPENMP),
      Rcpp::Named("max_threads") = omp_get_max_threads(),
      Rcpp::Named("num_procs")   = omp_get_num_procs(),
      Rcpp::Named("hw_threads")  = detectCoresCpp());
#else
  // Report 1 rather than the hardware count: without OpenMP the kernels run
  // serially whatever the caller passes, and saying so is the point.
  return Rcpp::List::create(
      Rcpp::Named("openmp")      = false,
      Rcpp::Named("spec")        = NA_INTEGER,
      Rcpp::Named("max_threads") = 1,
      Rcpp::Named("num_procs")   = 1,
      Rcpp::Named("hw_threads")  = detectCoresCpp());
#endif
}
