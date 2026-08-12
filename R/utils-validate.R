## Internal argument validators.
##
## These already exist as local closures inside sparseKNN() and (in the flag
## case) sparseSNN(). Adding a third literal copy for sparseKNNCross() is how
## the drift that this file exists to stop would continue, so the shared
## definitions live here instead.
##
## The bodies are semantically identical to the copies in sparseKNN(); those are
## deliberately left in place for now because they are covered by existing tests
## and changing them is a separate, independently reviewable commit. Once
## .knn_setup() is extracted, the local copies should be deleted in favour of
## these. Local definitions shadow these, so the duplication is inert until
## then.

#' @keywords internal
#' @noRd
.chk_flag <- function(value, name) {
  ## Strictly logical: isTRUE(1) is FALSE while Rcpp coerces 1 to TRUE, so a
  ## numeric flag would rank a distance as if it were a similarity.
  if (!is.logical(value) || length(value) != 1L || is.na(value)) {
    stop("'", name, "' must be TRUE or FALSE.")
  }
  invisible(TRUE)
}

#' @keywords internal
#' @noRd
.chk_count <- function(value, name, allow_zero = FALSE) {
  ## Reject anything that is not a single finite whole number rather than
  ## letting as.integer() truncate silently (block_size = 4.5), or turn a value
  ## into NA because it is non-finite (ncores = Inf) or above INT_MAX (k = 3e9),
  ## either of which would surface as an unrelated error further down.
  lower <- if (allow_zero) 0 else 1
  if (!is.numeric(value) || length(value) != 1L || is.na(value) ||
      !is.finite(value) || value != floor(value) || value < lower ||
      value > .Machine$integer.max) {
    stop("'", name, "' must be a ",
         if (allow_zero) "non-negative" else "positive",
         " integer no larger than .Machine$integer.max.")
  }
  as.integer(value)
}
