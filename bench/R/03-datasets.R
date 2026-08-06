## ---------------------------------------------------------------------------
## 03-datasets.R -- every matrix the benchmark runs on, behind one accessor.
##
##   get_dataset(id) -> list(X, density, signed, ...)
##
## THE CONTRACT. X always has OBSERVATIONS IN COLUMNS, matching sparseDist's
## orientation. Adapters that need rows (text2vec, BiocNeighbors, parallelDist)
## transpose in their prepare() step, where the cost is measured rather than
## hidden -- t() on a dgCMatrix is a full CSC rebuild, not a view.
##
## IDS ARE PARAMETRIC, not table lookups:
##
##   syn-n5000-p2000-d0.01-nonneg     synthetic, 2000 features x 5000 obs,
##                                    density 0.01, non-negative
##   syn-n5000-p2000-d0.01-signed     the same, with negative entries
##   pbmc-rna-hvg-n20000              scRNA-seq, log-normalised, 2000 HVGs
##   pbmc-rna-simplex-n20000          library-size normalised, columns sum to 1
##                                    (the only valid input for Jensen-Shannon)
##   pbmc-rna-pca50-n20000            50-PC embedding, DENSE
##   pbmc-atac-bin-n20000             scATAC-seq binary peak matrix
##
## so a density sweep or a subsample ladder is id generation, not a registry
## that can drift out of step with what the adapters expect.
##
## CACHING. Every cell runs in a fresh process, so a real dataset would be
## rebuilt hundreds of times without a disk cache. Builds are published with
## atomic_saveRDS(), which makes concurrent SLURM array tasks safe: several may
## build the same dataset at once, but each publishes atomically and the
## contents are identical.
## ---------------------------------------------------------------------------

## --- locations --------------------------------------------------------------

bench_data_dir <- function() {
  d <- Sys.getenv("BENCH_DATA", unset = "")
  if (!nzchar(d)) d <- file.path(Sys.getenv("HOME"), ".sparsedist-bench")
  d
}
raw_dir   <- function() file.path(bench_data_dir(), "raw")
cache_dir <- function() file.path(bench_data_dir(), "cache")

## --- raw sources ------------------------------------------------------------
##
## 10x publishes these as MatrixMarket, which Matrix::readMM() reads straight
## from a gzfile() connection -- so no Bioconductor data package is needed, and
## the Mac and the cluster are guaranteed byte-identical inputs rather than
## merely the same package version.
##
## !! THE URLS BELOW ARE UNVERIFIED. 10x reorganises its download paths, and
## !! these were written from the usual pattern rather than checked. Run
## !! verify_sources() once, on a machine with internet, before anything else:
## !! it fetches each file, records its sha256 in the manifest, and reports
## !! what failed. Fix any 404 by editing the url here -- then every later run,
## !! on either machine, verifies against the recorded checksum. A silently
## !! updated remote file would otherwise be undetectable.
BENCH_SOURCES <- list(
  pbmc_rna = list(
    description = "10x PBMC scRNA-seq, filtered feature-barcode matrix",
    url    = paste0("https://cf.10xgenomics.com/samples/cell-exp/3.0.0/",
                    "pbmc_10k_v3/pbmc_10k_v3_filtered_feature_bc_matrix.tar.gz"),
    archive = TRUE,
    sha256 = NA_character_          # recorded by verify_sources()
  ),
  pbmc_atac = list(
    description = "10x PBMC scATAC-seq, filtered peak-barcode matrix",
    url    = paste0("https://cf.10xgenomics.com/samples/cell-atac/1.0.1/",
                    "atac_v1_pbmc_10k/atac_v1_pbmc_10k_filtered_peak_bc_matrix.tar.gz"),
    archive = TRUE,
    sha256 = NA_character_
  )
)

sources_manifest_path <- function()
  file.path(bench_data_dir(), "sources-manifest.json")

read_sources_manifest <- function() {
  p <- sources_manifest_path()
  if (!file.exists(p)) return(list())
  tryCatch(jsonlite::fromJSON(p, simplifyVector = FALSE),
           error = function(e) list())
}

## Download (once) and return the local path. Verifies against the recorded
## checksum on every subsequent call.
fetch_source <- function(key, quiet = TRUE) {
  src <- BENCH_SOURCES[[key]]
  if (is.null(src)) stop("unknown source '", key, "'", call. = FALSE)

  dir.create(raw_dir(), recursive = TRUE, showWarnings = FALSE)
  dest <- file.path(raw_dir(), paste0(key, "-", basename(src$url)))

  if (!file.exists(dest)) {
    tmp <- paste0(dest, ".part")
    ok <- tryCatch({
      utils::download.file(src$url, tmp, mode = "wb", quiet = quiet)
      TRUE
    }, error = function(e) {stop("download failed for '", key, "': ",
                                 conditionMessage(e), call. = FALSE)})
    if (!file.rename(tmp, dest)) stop("could not finalise download: ", dest)
  }

  ## Checksum: recorded on first fetch, enforced thereafter. Without this a
  ## remote file changing under us would silently alter every result.
  man <- read_sources_manifest()
  recorded <- if (!is.null(man[[key]])) man[[key]]$sha256 else src$sha256
  actual <- digest::digest(file = dest, algo = "sha256")
  if (is.null(recorded) || is.na(recorded)) {
    man[[key]] <- list(sha256 = actual, url = src$url,
                       recorded_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ",
                                            tz = "UTC"))
    writeLines(jsonlite::toJSON(man, auto_unbox = TRUE, pretty = TRUE),
               sources_manifest_path())
  } else if (!identical(recorded, actual)) {
    stop("checksum mismatch for source '", key, "'.\n  recorded: ", recorded,
         "\n  actual:   ", actual,
         "\nThe remote file has changed, or the local copy is corrupt. ",
         "Delete ", dest, " to re-fetch.", call. = FALSE)
  }
  dest
}

## Fetch every source and report. Run once per machine before benchmarking.
verify_sources <- function() {
  for (key in names(BENCH_SOURCES)) {
    res <- tryCatch({p <- fetch_source(key, quiet = FALSE)
                     paste("OK  ", key, "->", basename(p))},
                    error = function(e) paste("FAIL", key, ":",
                                              conditionMessage(e)))
    cat(res, "\n")
  }
  invisible(NULL)
}

## Read a 10x matrix-market bundle into a dgCMatrix (features x cells).
read_10x <- function(key) {
  path <- fetch_source(key)
  ex <- file.path(raw_dir(), paste0(key, "-extracted"))
  if (!dir.exists(ex)) {
    dir.create(ex, recursive = TRUE, showWarnings = FALSE)
    utils::untar(path, exdir = ex)
  }
  mtx <- list.files(ex, pattern = "matrix\\.mtx(\\.gz)?$", recursive = TRUE,
                    full.names = TRUE)
  if (!length(mtx)) stop("no matrix.mtx found under ", ex, call. = FALSE)

  con <- if (grepl("\\.gz$", mtx[1])) gzfile(mtx[1], "rb") else file(mtx[1], "rb")
  on.exit(close(con), add = TRUE)
  X <- Matrix::readMM(con)
  methods::as(methods::as(X, "generalMatrix"), "CsparseMatrix")
}

## --- id parsing -------------------------------------------------------------

parse_dataset_id <- function(id) {
  if (grepl("^syn-", id)) {
    m <- regmatches(id, regexec(
      "^syn-n([0-9]+)-p([0-9]+)-d([0-9.]+)-(nonneg|signed|simplex)$",
      id))[[1]]
    if (!length(m)) {
      stop("malformed synthetic id '", id, "'; expected ",
           "syn-n<obs>-p<features>-d<density>-<nonneg|signed|simplex>",
           call. = FALSE)
    }
    return(list(kind = "syn", n_cols = as.integer(m[2]),
                n_rows = as.integer(m[3]), density = as.numeric(m[4]),
                sign = m[5], signed = identical(m[5], "signed"),
                simplex = identical(m[5], "simplex")))
  }
  m <- regmatches(id, regexec(
    "^pbmc-(rna|atac)-(hvg|simplex|pca[0-9]+|bin)-n([0-9]+)$", id))[[1]]
  if (!length(m)) stop("unrecognised dataset id '", id, "'", call. = FALSE)
  ## pca<N> carries its own dimensionality, which is what makes a controlled
  ## dimension sweep possible on real data with real cluster structure --
  ## synthetic uniform noise would be pathological for graph-based ANN and
  ## would overstate the effect.
  npcs <- if (grepl("^pca", m[3])) as.integer(sub("^pca", "", m[3])) else NA_integer_
  list(kind = "real", assay = m[2], form = m[3],
       is_pca = grepl("^pca", m[3]), npcs = npcs,
       n_cols = as.integer(m[4]))
}

## --- builders ---------------------------------------------------------------

## Deterministic seed from the id, so a dataset is identical on both machines
## and across reruns without the driver having to thread a seed through.
id_seed <- function(id) {
  h <- digest::digest(id, algo = "xxhash32", serialize = FALSE)
  as.integer(strtoi(substr(h, 1, 7), 16L) %% .Machine$integer.max)
}

build_synthetic <- function(id, spec) {
  set.seed(id_seed(id))
  X <- Matrix::rsparsematrix(spec$n_rows, spec$n_cols, density = spec$density)
  ## rsparsematrix draws from a symmetric distribution, so the default is
  ## signed; abs() gives the non-negative case. Both matter: cosine and
  ## correlation behave differently on signed data, and Jaccard/JS require
  ## non-negative input.
  if (!spec$signed) X <- abs(X)
  X <- methods::as(methods::as(X, "generalMatrix"), "CsparseMatrix")

  if (isTRUE(spec$simplex)) {
    ## Columns rescaled to relative frequencies -- the same operation the real
    ## simplex dataset performs on counts, and the only valid input for
    ## Jensen-Shannon. Kept SPARSE: normalising does not fill in zeros, and
    ## storing it dense would hide our own densification cost (see the note in
    ## build_real).
    ls <- Matrix::colSums(X)
    X <- X[, ls > 0, drop = FALSE]
    X <- methods::as(X %*% Matrix::Diagonal(x = 1 / Matrix::colSums(X)),
                     "CsparseMatrix")
  }
  colnames(X) <- paste0("c", seq_len(ncol(X)))
  X
}

## Column subsample, deterministic given the id.
subsample_cols <- function(X, n, id) {
  if (n >= ncol(X)) return(X)
  set.seed(id_seed(id))
  X[, sort(sample.int(ncol(X), n)), drop = FALSE]
}

## Drop all-zero rows. Features observed in no retained cell contribute nothing
## but inflate n_rows and depress the reported density, which would make the
## density axis of the sweep meaningless.
drop_empty_rows <- function(X) X[Matrix::rowSums(X != 0) > 0, , drop = FALSE]

## Counts -> log(CP10K + 1), the standard scRNA-seq normalisation.
log_normalise <- function(X, scale_factor = 1e4) {
  ls <- Matrix::colSums(X)
  ls[ls == 0] <- 1
  X <- X %*% Matrix::Diagonal(x = scale_factor / ls)
  X@x <- log1p(X@x)
  methods::as(X, "CsparseMatrix")
}

## Top-n most variable rows. Deliberately simple: this is a benchmark input,
## not a biological result, and a bespoke HVG method would only add a
## dependency and an argument.
top_variable_rows <- function(X, n) {
  if (nrow(X) <= n) return(X)
  mu <- Matrix::rowMeans(X)
  v  <- Matrix::rowMeans(X * X) - mu^2
  X[order(v, decreasing = TRUE)[seq_len(n)], , drop = FALSE]
}

build_real <- function(id, spec) {
  key <- if (identical(spec$assay, "rna")) "pbmc_rna" else "pbmc_atac"
  X <- read_10x(key)
  colnames(X) <- paste0("c", seq_len(ncol(X)))
  X <- subsample_cols(X, spec$n_cols, id)
  X <- drop_empty_rows(X)

  if (identical(spec$form, "bin")) {
    ## scATAC: the standard representation is presence/absence of a peak, which
    ## is exactly what method = "binary" consumes -- and the sparsest regime in
    ## the whole benchmark, often under 1%.
    X@x <- rep(1, length(X@x))
    return(X)
  }
  if (identical(spec$form, "hvg")) {
    ## Note the density here CLIMBS relative to raw counts -- HVG selection
    ## keeps the busiest features -- so this sits at the unfavourable end of
    ## the sparse/dense crossover, and reporting that is the point.
    return(top_variable_rows(log_normalise(X), 2000L))
  }
  if (identical(spec$form, "simplex")) {
    ## Jensen-Shannon needs non-negative columns summing to 1. NOT log
    ## transformed: log-normalised values are not a probability distribution,
    ## and feeding them to JS would silently compute something meaningless.
    ls <- Matrix::colSums(X)
    keep <- ls > 0
    X <- X[, keep, drop = FALSE]
    X <- X %*% Matrix::Diagonal(x = 1 / Matrix::colSums(X))
    X <- top_variable_rows(methods::as(X, "CsparseMatrix"), 2000L)
    ## Renormalise after feature selection, or the columns no longer sum to 1.
    ## (Selecting then normalising gives the same matrix; the order only
    ## matters for which intermediate is materialised.)
    cs <- Matrix::colSums(X); cs[cs == 0] <- 1
    ## Returned SPARSE, not dense, even though the js kernel needs a dense
    ## arma::mat. The sparseDist adapter densifies in prepare(), where the cost
    ## is measured; storing it dense instead would hide that cost from us AND
    ## charge proxyC for a dense -> CSC conversion it would never do in
    ## practice. Exactly the wrong way round.
    return(methods::as(X %*% Matrix::Diagonal(x = 1 / cs), "CsparseMatrix"))
  }
  if (isTRUE(spec$is_pca)) {
    ## The DENSE case, and deliberately the same biology rather than unrelated
    ## data: it is a fair baseline for parallelDist, it is the input
    ## BiocNeighbors is actually designed for (Seurat and scanpy both cluster
    ## here, not on the count matrix), and at density 1.0 it is where our CSC
    ## merge walk should lose to a contiguous dense loop. Reporting where the
    ## package should NOT be used is stronger than omitting it.
    if (!requireNamespace("irlba", quietly = TRUE)) {
      stop("dataset '", id, "' needs the 'irlba' package; add it to setup.R.",
           call. = FALSE)
    }
    H <- top_variable_rows(log_normalise(X), 2000L)
    set.seed(id_seed(id))
    nv <- spec$npcs
    if (nv >= min(dim(H))) {
      stop("dataset '", id, "': ", nv, " components requested but the matrix ",
           "is ", nrow(H), " x ", ncol(H), call. = FALSE)
    }
    pc <- irlba::irlba(Matrix::t(H), nv = nv, center = Matrix::rowMeans(H))
    E <- t(pc$u %*% diag(pc$d))                # nv x n_cells, dense
    colnames(E) <- colnames(X)
    return(E)
  }
  stop("unhandled form '", spec$form, "'", call. = FALSE)
}

## --- accessor ---------------------------------------------------------------

describe_dataset <- function(id, X) {
  nz <- if (methods::is(X, "sparseMatrix")) Matrix::nnzero(X) else sum(X != 0)
  list(
    id       = id,
    X        = X,
    n_rows   = nrow(X),
    n_cols   = ncol(X),
    density  = nz / (as.numeric(nrow(X)) * as.numeric(ncol(X))),
    signed   = any(X < 0),
    ## Adapters branch on these rather than re-deriving them, so that a
    ## competitor is never handed input it cannot legitimately consume --
    ## Jaccard on non-binary data, or JS on columns that do not sum to 1.
    ## Sparse and dense need different tests, and the order matters: X@x on a
    ## base matrix is an ERROR, not a fallback, so evaluating it first would
    ## make every dense dataset (pca50) fail to load.
    binary   = if (methods::is(X, "sparseMatrix")) isTRUE(all(X@x == 1))
               else isTRUE(all(X %in% c(0, 1))),
    simplex  = isTRUE(all(abs(Matrix::colSums(X) - 1) < 1e-8)),
    is_sparse = methods::is(X, "sparseMatrix")
  )
}

get_dataset <- function(id, use_cache = TRUE) {
  cf <- file.path(cache_dir(), paste0(id, ".rds"))
  if (use_cache && file.exists(cf)) {
    out <- tryCatch(readRDS(cf), error = function(e) NULL)
    if (!is.null(out)) return(out)
    unlink(cf)                       # truncated cache entry; rebuild
  }

  spec <- parse_dataset_id(id)
  X <- if (identical(spec$kind, "syn")) build_synthetic(id, spec)
       else                             build_real(id, spec)
  out <- describe_dataset(id, X)

  if (use_cache) {
    ## Atomic: concurrent array tasks may build the same dataset at once, and
    ## a half-written cache file read by a sibling would be far worse than the
    ## duplicated work.
    dir.create(cache_dir(), recursive = TRUE, showWarnings = FALSE)
    tryCatch(atomic_saveRDS(out, cf, compress = FALSE),
             error = function(e) warning("could not cache '", id, "': ",
                                         conditionMessage(e), call. = FALSE))
  }
  out
}

## --- id generators for the drivers -----------------------------------------

## Density sweep at fixed shape. Spans the crossover: our per-pair merge walk
## costs O(p^2 * nnz-per-column) while proxyC's and text2vec's matrix products
## skip disjoint pairs entirely, so there is a density at which each wins.
## Finding it deliberately turns a potential weakness into a characterisation.
ids_density_sweep <- function(n_cols = 3000, n_rows = 2000,
                              densities = c(0.5, 0.1, 0.05, 0.01, 0.005,
                                            0.001),
                              sign = "nonneg") {
  sprintf("syn-n%d-p%d-d%s-%s", n_cols, n_rows,
          format(densities, scientific = FALSE, trim = TRUE), sign)
}

## Subsample ladder for the scaling and frontier experiments.
ids_size_ladder <- function(prefix = "pbmc-rna-hvg",
                            n = c(1000, 5000, 20000, 50000, 100000, 200000)) {
  sprintf("%s-n%d", prefix, n)
}

## Simplex variants of the same sweep, for Jensen-Shannon.
##
## Worth running even though our js kernel densifies and is therefore
## density-INSENSITIVE: proxyC's "jensen" works on CSC and should get faster as
## density falls, so there is a crossover here that we lose. Measuring it is
## the point.
ids_density_sweep_simplex <- function(n_cols = 3000, n_rows = 2000,
                                      densities = c(0.5, 0.1, 0.05, 0.01,
                                                    0.005, 0.001)) {
  ids_density_sweep(n_cols, n_rows, densities, sign = "simplex")
}

## Everything the alignment suite needs: small, fast, and covering each
## kernel's admissible input.
ids_alignment <- function() {
  c("syn-n300-p500-d0.1-nonneg",
    "syn-n300-p500-d0.1-signed",     # cosine/correlation differ on signed data
    "syn-n300-p500-d0.01-nonneg",    # degenerate columns appear at low density
    "pbmc-rna-hvg-n1000",
    "pbmc-rna-simplex-n1000",        # JS
    "pbmc-rna-pca50-n1000",          # dense
    "pbmc-atac-bin-n1000")           # binary / Jaccard
}
