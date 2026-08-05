// Detect the number of concurrent threads supported by the hardware.
//
// Approach borrowed from RcppThread:
//   https://github.com/tnagler/RcppThread/commit/c26fc2b0d56555fa434c33352747822691334fe8
// -----------------------------------------------------------------------------

#include <Rcpp.h>

#include <thread>
#include <climits>

// Number of concurrent threads supported by the hardware.
// Returns at least 1: hardware_concurrency() yields 0 when the value is not
// computable, and a 0 thread count is never a useful answer for a caller.
//
// [[Rcpp::export]]
int detectCoresCpp() {
  const unsigned int n = std::thread::hardware_concurrency();
  if (n == 0u) return 1;
  if (n > static_cast<unsigned int>(INT_MAX)) return INT_MAX;
  return static_cast<int>(n);
}
