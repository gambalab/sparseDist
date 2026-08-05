## Test environments

* local macOS Sonoma 14.7, R 4.6.1 (x86_64)
* win-builder (devel and release)
* R-hub: Windows Server, Ubuntu Linux, macOS

## R CMD check results

0 errors | 0 warnings | 0 notes

This is a new release.

## Notes for the reviewer

* **Parallelism.** The C++ kernels use OpenMP where the compiler provides it,
  guarded by `#ifdef _OPENMP`, and fall back to serial execution otherwise.
  The thread count is controlled by the `ncores` argument of every exported
  function and defaults to 1. All examples and tests pass `ncores = 1`
  explicitly, so no check ever uses more than a single core. Passing
  `ncores = 0` opts in to auto-detection, and one test that exercises that path
  mocks the detection function so it still resolves to 1.

* **C++ standard.** No `CXX_STD` is set and `SystemRequirements` names no
  standard: the package needs only C++11 features, which is below the current
  default on every supported R version.

* **Long-running examples.** All examples use small matrices and complete in
  well under a second.
