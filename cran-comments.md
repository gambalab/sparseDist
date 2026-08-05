## Test environments

* local macOS 14, R 4.6.0
* win-builder (devel and release)
* R-hub: Windows Server, Ubuntu Linux, macOS

## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new release.

## Notes

* The package uses OpenMP where the compiler provides it. Thread count is
  controlled by the `ncores` argument, which defaults to 1 (serial). Examples
  and tests pass `ncores = 1`, so no check ever uses more than one core.
