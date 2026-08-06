## ---------------------------------------------------------------------------
## 00-utils.R -- small shared helpers.
## ---------------------------------------------------------------------------

`%||%` <- function(a, b) if (is.null(a)) b else a

## Publish an object at `path` only once it is completely written.
##
## Both parent and child write result rows, and either can be killed mid-write
## (timeout, OOM, pre-emption). A half-written .rds at the final path is worse
## than no file: it exists, so the parent treats the cell as done, and readRDS()
## then fails or -- worse -- succeeds on a truncated object.
##
## The temporary must be in the SAME directory: rename() is atomic only within
## a filesystem, and tempdir() is frequently a different mount on a cluster.
## Note this makes the temporary name roughly 20 characters longer than the
## final one, which is why make_cell_id() budgets for it.
atomic_saveRDS <- function(object, path, compress = TRUE) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  tmp <- tempfile(pattern = paste0(".", basename(path), "."),
                  tmpdir = dirname(path))
  on.exit(unlink(tmp), add = TRUE)

  saveRDS(object, tmp, compress = compress)

  ## On POSIX rename() replaces atomically, so unlinking first would only open
  ## a window in which neither file is at `path`. Windows rename fails on an
  ## existing target, so there it must go.
  if (.Platform$OS.type == "windows" && file.exists(path)) unlink(path)
  if (!file.rename(tmp, path)) {
    stop("Could not atomically publish result: ", path, call. = FALSE)
  }
  invisible(path)
}

## --- scalar predicates -----------------------------------------------------

is_whole_scalar <- function(x, min = 1, max = .Machine$integer.max) {
  is.numeric(x) && length(x) == 1L && !is.na(x) && is.finite(x) &&
    x == floor(x) && x >= min && x <= max
}

## "Not supplied" -- covering both an absent field and an explicit NA. These
## are deliberately equivalent everywhere in the harness: new_cell_spec()
## normalises the former into the latter, so no downstream code needs to know
## which one the driver wrote.
is_absent <- function(x) is.null(x) || (length(x) == 1L && is.na(x))

## --- cell identity ---------------------------------------------------------

## Canonicalise a spec's identity-bearing fields, VALIDATING as it goes.
##
## Validation belongs here, not only in validate_spec(), because of the call
## order: make_cell_id() necessarily runs FIRST -- validate_spec() checks
## cell_id, which does not exist until make_cell_id() has produced it. So this
## is the only place a malformed value can be caught before it has already
## become some other cell's filename.
##
## The specific hazard is lossy coercion. as.integer(2.5) is 2, so a malformed
## k = 2.5 would canonicalise identically to a valid k = 2, receive the same
## id, and overwrite that cell's result. Same for block_size. Rejecting is the
## only safe response: silently rounding would fabricate a cell that was never
## requested.
##
## Canonicalisation also serves the converse purpose -- threads = 1 and
## threads = 1L, or an omitted variant and an explicit NA_character_, are one
## logical cell and must hash to one id, or a plot gains duplicate points.
canonical_identity <- function(spec) {
  required_int <- function(x, name, min = 1) {
    if (!is_whole_scalar(x, min = min)) {
      stop("'", name, "' must be a whole number between ", min,
           " and .Machine$integer.max; got ",
           paste(format(x), collapse = ","), call. = FALSE)
    }
    as.integer(x)
  }
  optional_int <- function(x, name, min = 1) {
    if (is_absent(x)) return(NA_integer_)
    required_int(x, name, min)
  }
  chr_or_na <- function(x) {
    if (is_absent(x) || length(x) != 1L) NA_character_ else as.character(x)
  }

  list(experiment = chr_or_na(spec$experiment),
       package    = chr_or_na(spec$package),
       method     = chr_or_na(spec$method),
       dataset_id = chr_or_na(spec$dataset_id),
       threads    = required_int(spec$threads, "threads"),
       phase      = chr_or_na(spec$phase),
       rep        = required_int(spec$rep, "rep"),
       seed       = optional_int(spec$seed, "seed",
                                 min = -.Machine$integer.max),
       k          = optional_int(spec$k, "k"),
       block_size = optional_int(spec$block_size, "block_size"),
       variant    = chr_or_na(spec$variant))
}

## Deterministic, filesystem-safe, collision-RESISTANT cell identifier.
##
## The readable prefix is for humans scanning a results directory; the hash is
## what guarantees distinctness. Both are needed:
##
##   - a prefix alone is NOT unique. It omits block_size and seed (both
##     first-class), so two kNN cells differing only in block size, or two
##     cells generating different synthetic data, would share a filename and
##     one result would silently overwrite the other.
##   - sanitisation collapses genuinely distinct values: the variants
##     "dist=FALSE/full=TRUE" and "dist=FALSE full=TRUE" map to the same safe
##     string. Length-triggered hashing would not catch it; these ids are short.
##
## 128 bits of sha1. Not "collision-proof" -- a truncated hash never is -- but
## a collision would overwrite a result, so the margin is worth the length.
CELL_IDENTITY_FIELDS <- c("experiment", "package", "method", "dataset_id",
                          "threads", "phase", "rep", "seed", "k",
                          "block_size", "variant")

make_cell_id <- function(spec) {
  identity <- canonical_identity(spec)

  readable <- paste(spec$experiment, spec$package, spec$method,
                    spec$dataset_id, paste0("t", spec$threads), spec$phase,
                    paste0("r", spec$rep), sep = "-")
  readable <- gsub("[^A-Za-z0-9_.-]+", "_", readable)

  ## 140 + 1 + 32 = 173 characters, leaving room for ".rds" and the ~20 extra
  ## characters atomic_saveRDS's temporary name adds, under the 255-byte limit.
  hash <- substr(digest::digest(identity, algo = "sha1", serialize = TRUE),
                 1, 32)
  paste0(substr(readable, 1, 140), "-", hash)
}

## --- spec construction and validation --------------------------------------

## Fields that must be non-empty character scalars.
##
## Type matters here, not merely presence. The parent compares identifiers to
## the spec with identical(), while the child round-trips them through
## as.character() in new_result_row(). A numeric run_id = 123 therefore yields
## a child row holding "123", which the parent rejects as belonging to a
## different cell -- so it discards a perfectly good result and synthesises a
## failure row in its place. Silently, for every cell.
SPEC_TEXT_FIELDS <- c("run_id", "cell_id", "experiment", "package", "method",
                      "dataset_id", "phase")

## The ONLY supported way to build a cell spec.
##
## Normalising optional fields to NA at construction removes what would
## otherwise be an implicit contract: because an omitted k and an explicit
## NA k hash to the same id, they must also behave identically in adapters.
## Rather than documenting that and hoping, this guarantees adapters never
## receive NULL for an optional field -- there is nothing left to get wrong.
new_cell_spec <- function(run_id, experiment, package, method, dataset_id,
                          threads, phase, rep,
                          k = NA, block_size = NA, variant = NA, seed = NA,
                          ...) {
  spec <- list(run_id = run_id, experiment = experiment, package = package,
               method = method, dataset_id = dataset_id,
               threads = threads, phase = phase, rep = rep,
               k          = if (is_absent(k))          NA else k,
               block_size = if (is_absent(block_size)) NA else block_size,
               variant    = if (is_absent(variant))    NA else variant,
               seed       = if (is_absent(seed))       NA else seed)
  extra <- list(...)
  if (length(extra)) spec[names(extra)] <- extra

  ## canonical_identity() inside make_cell_id() validates the numeric fields;
  ## validate_spec() then covers the textual and vocabulary constraints.
  spec$cell_id <- make_cell_id(spec)
  validate_spec(spec)
  spec
}

validate_spec <- function(spec) {
  ## Textual identity first: the vocabulary checks below index into these, and
  ## a %in% test against a factor or number gives a confusing error rather than
  ## a useful one.
  bad_text <- SPEC_TEXT_FIELDS[!vapply(SPEC_TEXT_FIELDS, function(nm) {
    x <- spec[[nm]]
    is.character(x) && length(x) == 1L && !is.na(x) && nzchar(x)
  }, logical(1))]
  if (length(bad_text)) {
    stop("Cell spec requires non-empty character field(s): ",
         paste(bad_text, collapse = ", "), call. = FALSE)
  }

  ## A typo here ("kernal") would silently select the end-to-end branch and
  ## fold coercion cost into what is reported as a kernel timing.
  if (!spec$phase %in% PHASE_LEVELS) {
    stop("Invalid phase '", spec$phase, "'; expected one of: ",
         paste(PHASE_LEVELS, collapse = ", "), call. = FALSE)
  }
  if (!spec$experiment %in% EXPERIMENT_LEVELS) {
    stop("Invalid experiment '", spec$experiment, "'; expected one of: ",
         paste(EXPERIMENT_LEVELS, collapse = ", "), call. = FALSE)
  }

  ## Bounded above by integer.max: as.integer(3e9) is NA, and a downstream
  ## `if (NA)` fails with "missing value where TRUE/FALSE needed" instead of
  ## the intended message.
  if (!is_whole_scalar(spec$threads, min = 1)) {
    stop("threads must be a positive whole number no larger than ",
         ".Machine$integer.max; got ",
         paste(format(spec$threads), collapse = ","), call. = FALSE)
  }
  if (!is_whole_scalar(spec$rep, min = 1)) {
    stop("rep must be a positive whole number; got ",
         paste(format(spec$rep), collapse = ","), call. = FALSE)
  }
  ## Optional numerics: absent (or NA, which new_cell_spec normalises to) is
  ## fine, but a supplied value must be a whole number -- set.seed() on a
  ## fractional seed silently truncates, breaking reproducibility with no error.
  for (nm in c("seed", "k", "block_size")) {
    x <- spec[[nm]]
    if (is_absent(x)) next
    lo <- if (identical(nm, "seed")) -.Machine$integer.max else 1
    if (!is_whole_scalar(x, min = lo)) {
      stop("'", nm, "', when supplied, must be a whole number within integer ",
           "range; got ", paste(format(x), collapse = ","), call. = FALSE)
    }
  }

  ## cell_id becomes a filename, so it must not contain path separators.
  if (!grepl("^[A-Za-z0-9_.-]+$", spec$cell_id)) {
    stop("Unsafe cell_id (filename-forming): ", spec$cell_id, call. = FALSE)
  }
  invisible(spec)
}

## --- scheduler provenance --------------------------------------------------

## SLURM identifiers, so post-hoc `sacct` output can be joined to result rows.
##
## Without these, an OOM confirmed from scheduler accounting has no key to join
## on -- which is why the harness records "killed" and leaves the OOM
## determination to an external annotation rather than guessing at write time.
slurm_provenance <- function() {
  get <- function(v) {
    x <- Sys.getenv(v, unset = "")
    if (nzchar(x)) x else NA_character_
  }
  list(slurm_job_id        = get("SLURM_JOB_ID"),
       slurm_array_job_id  = get("SLURM_ARRAY_JOB_ID"),
       slurm_array_task_id = get("SLURM_ARRAY_TASK_ID"))
}
