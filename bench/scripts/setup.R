#!/usr/bin/env Rscript
## ---------------------------------------------------------------------------
## setup.R -- install and pin every package the benchmark needs, then prove the
## environment is actually usable.
##
## Run on the Mac first, then on the cluster with the SAME constants below. The
## point is not merely that the packages install, but that both machines end up
## with the same VERSIONS -- otherwise a discrepancy between the dry run and the
## real run is unattributable.
##
##   Rscript bench/scripts/setup.R --lib=../sparseDist-bench-lib
##
## KEEP THE LIBRARY OUTSIDE THE PACKAGE DIRECTORY. A library inside the package
## root makes R CMD build try to ship several hundred megabytes of Boost and
## Eigen headers, and makes git, RStudio indexing and devtools::check() walk
## them on every invocation. If you do keep it inside, add ^bench-lib$ to
## .Rbuildignore.
##
## Installation is the slowest step on a cluster, so this prefers a binary
## repository where one is available. Source builds of text2vec, dbscan and the
## Bioconductor stack take a long time on a cold node.
## ---------------------------------------------------------------------------

## --- pins ------------------------------------------------------------------
## A dated CRAN snapshot is what makes "same versions on both machines" true
## without hand-maintaining a version list. Set this once, change it never
## (until the paper is accepted).
CRAN_SNAPSHOT <- "2026-08-01"        # <-- set to the date you first install
BIOC_VERSION  <- NULL                 # NULL = whatever matches this R; else e.g. "3.21"

## The adapter layer was written and verified against R 4.6.1 with
## BiocNeighbors 2.6.0, Matrix 1.7.5 and proxyC 0.5.2. R 4.2 pins Bioconductor
## 3.16, whose BiocNeighbors is the 1.x line with a DIFFERENT API, and ships a
## Matrix predating the generalMatrix coercion path as_dgc() relies on. A run
## on that stack would not be the benchmark we verified.
MIN_R      <- "4.4.0"
VERIFIED_R <- "4.6.1"

args    <- commandArgs(trailingOnly = TRUE)
lib_arg <- grep("^--lib=", args, value = TRUE)
LIB     <- if (length(lib_arg)) sub("^--lib=", "", lib_arg[1]) else .libPaths()[1]
dir.create(LIB, recursive = TRUE, showWarnings = FALSE)
LIB <- normalizePath(LIB, mustWork = FALSE)
.libPaths(c(LIB, .libPaths()))

## Bench root, derived from this script's own path so the manifest lands in
## bench/results/ regardless of the working directory.
script_path <- function() {
  ca  <- commandArgs(trailingOnly = FALSE)
  hit <- grep("^--file=", ca, value = TRUE)
  if (length(hit)) normalizePath(sub("^--file=", "", hit[1]), mustWork = FALSE)
  else NA_character_
}
sp         <- script_path()
BENCH_ROOT <- Sys.getenv("BENCH_ROOT",
                         unset = if (is.na(sp)) getwd() else dirname(dirname(sp)))

say  <- function(...) cat(sprintf(...), "\n", sep = "")
line <- function(label, ok, detail = "")
  cat(sprintf("  %-34s %-6s %s\n", label, if (ok) "PASS" else "FAIL", detail))

if (getRversion() < MIN_R && !nzchar(Sys.getenv("BENCH_ALLOW_OLD_R"))) {
  stop("R >= ", MIN_R, " is required; this is ", getRversion(), ".\n",
       "The competitor APIs were verified on R ", VERIFIED_R, ". On an older ",
       "R, BiocManager pins an older Bioconductor and BiocNeighbors resolves ",
       "to the 1.x line, whose findKNN() signature differs from the 2.x one ",
       "the adapters are written against.\n",
       "Load a newer R module, or build one (conda/mamba), before running ",
       "this. Set BENCH_ALLOW_OLD_R=1 to override -- but then re-run ",
       "api-dump.R, api-probe2.R and align.R on THIS stack, because none of ",
       "the recorded semantics can be assumed to hold.",
       call. = FALSE)
}
if (getRversion() < VERIFIED_R) {
  say("NOTE: R %s here vs %s where the APIs were verified. Check align.R ",
      as.character(getRversion()), VERIFIED_R)
  say("      passes on this machine before trusting any result.")
}

## --- repository ------------------------------------------------------------
## Posit Package Manager serves dated snapshots, and on supported Linux distros
## also serves precompiled binaries -- the difference between minutes and hours
## on a cold cluster node. macOS falls back to the source snapshot, which is
## fine because the Mac install happens once and is not on the critical path.
ppm_repo <- function(snapshot) {
  base <- paste0("https://packagemanager.posit.co/cran/", snapshot)
  if (Sys.info()[["sysname"]] != "Linux") return(base)
  os <- tryCatch({
    rel <- readLines("/etc/os-release", warn = FALSE)
    get1 <- function(key) {
      hit <- grep(paste0("^", key, "="), rel, value = TRUE)
      if (!length(hit)) "" else gsub('"', '', sub(paste0("^", key, "="), "", hit[1]))
    }
    list(id = get1("ID"), codename = get1("VERSION_CODENAME"),
         version = get1("VERSION_ID"))
  }, error = function(e) list(id = "", codename = "", version = ""))

  ## Only a few distro strings are served; anything else must use source, and
  ## saying so is better than silently 404-ing every install.
  tag <- switch(os$id,
    ubuntu = os$codename,                                   # jammy, noble, ...
    debian = os$codename,                                   # bookworm, ...
    rhel   = paste0("rhel", sub("\\..*", "", os$version)),
    centos = paste0("centos", sub("\\..*", "", os$version)),
    rocky  = paste0("rhel", sub("\\..*", "", os$version)),
    almalinux = paste0("rhel", sub("\\..*", "", os$version)),
    "")
  if (!nzchar(tag)) {
    say("NOTE: distro '%s' has no binary channel; building from source.", os$id)
    return(base)
  }
  say("Binary repository: %s / %s", os$id, tag)
  paste0("https://packagemanager.posit.co/cran/__linux__/", tag, "/", snapshot)
}

repo <- ppm_repo(CRAN_SNAPSHOT)
options(repos = c(CRAN = repo),
        Ncpus = max(1L, parallel::detectCores() - 1L),
        warn  = 1)
say("CRAN snapshot: %s", repo)
say("Library:       %s", LIB)
say("Bench root:    %s", BENCH_ROOT)

## --- CRAN packages ---------------------------------------------------------
cran_pkgs <- c(
  ## competitors
  "proxyC",        # closest sparse pairwise competitor
  "text2vec",      # sparse cosine / Jaccard, row-oriented
  "coop",          # cosine, pcor, covar specialist
  "parallelDist",  # multithreaded dense baseline
  "philentropy",   # Jensen-Shannon reference (divergence, log2 by default)
  "dbscan",        # shared-COUNT SNN; workflow comparator only
  ## infrastructure
  "Matrix", "Rcpp", "RcppArmadillo", "RcppProgress", "RcppParallel",
  "processx",      # wall-clock cap for benchmark cells
  "RhpcBLASctl",   # pin BLAS threads at run time as well as via env
  "jsonlite", "digest",
  "igraph",        # adjacency extraction for the bluster SNN comparison
  "irlba",         # 50-PC embedding for the dense dataset (pca50)
  ## data wrangling / reporting
  "data.table", "ggplot2", "scales",
  ## package tooling
  "remotes", "testthat", "BiocManager"
)

## Presence must be checked IN THE PINNED LIBRARY, not anywhere on the search
## path.
##
## requireNamespace() searches every .libPaths() entry, so an old copy in the
## system library counts as "already present" and nothing gets installed --
## which silently defeats the whole point of a pinned snapshot. It bit us on
## the cluster: a stale BiocManager in the system library reported Bioconductor
## 3.12 and the install failed with a version error that looked like a
## Bioconductor problem rather than a library-path one.
in_lib <- function(p) {
  length(find.package(p, lib.loc = LIB, quiet = TRUE)) > 0L
}

need <- cran_pkgs[!vapply(cran_pkgs, in_lib, logical(1))]
if (length(need)) {
  say("Installing %d CRAN package(s): %s", length(need), paste(need, collapse = ", "))
  install.packages(need, lib = LIB)
} else say("All CRAN packages already present.")

## --- Bioconductor ----------------------------------------------------------
## BiocManager pins versions by RELEASE, so no snapshot date is needed -- but
## the release must match this R, and it will refuse otherwise. That refusal is
## the single most common cluster install failure; let it happen here, loudly,
## rather than inside a queued job.
bioc_pkgs <- c("BiocNeighbors",   # exact + approximate kNN
               "bluster")         # neighborsToSNNGraph(type = "jaccard")

if (!in_lib("BiocManager")) {
  stop("BiocManager did not install into ", LIB, "; cannot continue.")
}
## Drop any already-loaded (system) BiocManager so the freshly installed one is
## used. Otherwise version() still answers from the stale copy.
if ("BiocManager" %in% loadedNamespaces()) {
  try(unloadNamespace("BiocManager"), silent = TRUE)
}
loadNamespace("BiocManager", lib.loc = LIB)
bioc_ver <- if (is.null(BIOC_VERSION)) BiocManager::version() else BIOC_VERSION
say("Bioconductor release: %s", as.character(bioc_ver))

need_bioc <- bioc_pkgs[!vapply(bioc_pkgs, in_lib, logical(1))]
if (length(need_bioc)) {
  ## BiocManager wants its own repo set; restore ours afterwards so the CRAN
  ## snapshot pin is not silently lost for anything installed later.
  old_repos <- getOption("repos")
  BiocManager::install(need_bioc, version = bioc_ver, lib = LIB,
                       ask = FALSE, update = FALSE)
  options(repos = old_repos)
} else say("All Bioconductor packages already present.")

## --- sparseDist itself -----------------------------------------------------
## Installed from the local source tree, not from a repository: the whole point
## is to benchmark THIS working copy.
pkg_src <- Sys.getenv("SPARSEDIST_SRC", unset = dirname(BENCH_ROOT))
has_omp_flags <- NA
if (file.exists(file.path(pkg_src, "DESCRIPTION"))) {
  say("Installing sparseDist from %s", normalizePath(pkg_src, mustWork = FALSE))

  ## Verify the OpenMP build flags BEFORE installing. Rcpp::plugins(openmp) is
  ## honoured only by sourceCpp(); a package build needs src/Makevars, and
  ## without it every #pragma omp in the package is silently ignored on every
  ## platform -- ncores would do nothing and the scaling experiment would
  ## measure noise.
  mk <- file.path(pkg_src, "src", "Makevars")
  has_omp_flags <- file.exists(mk) &&
    any(grepl("SHLIB_OPENMP_CXXFLAGS", readLines(mk, warn = FALSE)))
  if (!has_omp_flags) {
    say("")
    say("!! src/Makevars is missing or does not reference SHLIB_OPENMP_CXXFLAGS.")
    say("!! sparseDist will build WITHOUT OpenMP and ncores will have no effect.")
    say("!! Add src/Makevars containing:")
    say("!!     PKG_CXXFLAGS = $(SHLIB_OPENMP_CXXFLAGS)")
    say("!!     PKG_LIBS     = $(SHLIB_OPENMP_CXXFLAGS) $(LAPACK_LIBS) $(BLAS_LIBS) $(FLIBS)")
    say("")
  }
  remotes::install_local(pkg_src, lib = LIB, upgrade = "never", force = TRUE)
} else {
  say("NOTE: no DESCRIPTION at '%s'; set SPARSEDIST_SRC.", pkg_src)
}

## --- verification ----------------------------------------------------------
say("")
say("== verification ==")
all_pkgs <- c(cran_pkgs, bioc_pkgs, "sparseDist")
versions <- character(0)
for (p in all_pkgs) {
  ## Report the version FROM THE PINNED LIBRARY and say so when a different
  ## copy is shadowing it elsewhere on the path -- that is the situation this
  ## whole script exists to make impossible.
  here <- in_lib(p)
  v <- if (here) as.character(utils::packageVersion(p, lib.loc = LIB))
       else NA_character_
  versions[p] <- v
  other <- tryCatch(as.character(utils::packageVersion(p)),
                    error = function(e) NA_character_)
  detail <- if (here) {
    if (!is.na(other) && !identical(other, v))
      paste0(v, "  (WARNING: ", other, " also on the search path)") else v
  } else if (!is.na(other)) {
    paste0("NOT IN ", basename(LIB), " (", other, " found elsewhere)")
  } else "NOT INSTALLED"
  line(p, here, detail)
}

omp <- tryCatch(sparseDist:::ompInfoCpp(), error = function(e) NULL)
if (!is.null(omp)) {
  ## Diagnose the CAUSE rather than always blaming the Makevars. When the build
  ## flags are present and OpenMP is still absent, the toolchain is the reason
  ## -- stock Apple clang ships no OpenMP at all -- and sending someone hunting
  ## through a correct src/Makevars wastes their afternoon.
  detail <- if (isTRUE(omp$openmp)) {
    sprintf("spec %s, %s procs of %s hw threads",
            omp$spec, omp$num_procs, omp$hw_threads)
  } else if (isTRUE(has_omp_flags)) {
    "absent: build flags are correct, so this toolchain has no OpenMP (expected on stock macOS clang) -- correctness only, no timings"
  } else if (identical(has_omp_flags, FALSE)) {
    "INERT -- see the src/Makevars note above"
  } else {
    "absent, and src/Makevars was not checked (sparseDist not built here)"
  }
  line("sparseDist OpenMP", isTRUE(omp$openmp), detail)
}

## --- manifest --------------------------------------------------------------
## Written into bench/results/ so any figure can be traced to the exact
## environment that produced it, alongside the data it describes.
manifest <- list(
  written_at    = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  cran_snapshot = CRAN_SNAPSHOT,
  repo          = repo,
  bioc_version  = as.character(bioc_ver),
  r_version     = as.character(getRversion()),
  platform      = R.version$platform,
  host          = Sys.info()[["nodename"]],
  os            = paste(Sys.info()[["sysname"]], Sys.info()[["release"]]),
  blas          = tryCatch(utils::sessionInfo()$BLAS, error = function(e) NA),
  library       = LIB,
  openmp        = if (is.null(omp)) NA else isTRUE(omp$openmp),
  makevars_omp  = has_omp_flags,
  packages      = as.list(versions)
)
res_dir <- file.path(BENCH_ROOT, "results")
dir.create(res_dir, recursive = TRUE, showWarnings = FALSE)
out <- file.path(res_dir,
                 paste0("bench-manifest-", Sys.info()[["nodename"]], ".json"))
writeLines(jsonlite::toJSON(manifest, auto_unbox = TRUE, pretty = TRUE), out)
say("")
say("Manifest written to %s", out)

failed <- names(versions)[is.na(versions)]
if (length(failed)) {
  say("")
  stop("Setup incomplete; missing: ", paste(failed, collapse = ", "))
}
say("Setup complete.")
say("Next: Rscript %s/scripts/preflight.R", BENCH_ROOT)
