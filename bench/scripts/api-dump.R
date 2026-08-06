#!/usr/bin/env Rscript
## ---------------------------------------------------------------------------
## api-dump.R -- record the exact signatures and default behaviour of every
## competitor, from the versions actually installed.
##
##   Rscript bench/scripts/api-dump.R > bench/results/api-dump.txt 2>&1
##
## WHY THIS EXISTS. The adapters encode semantic decisions -- whether Jaccard
## is computed on values or on the sparsity pattern, whether output is rounded,
## how a directed neighbour graph becomes undirected. Get one wrong and the
## benchmark produces a plausible NUMBER rather than an error, which no amount
## of schema validation will catch. Writing adapters against remembered APIs is
## how that happens, so: read the installed reality first.
##
## Everything is wrapped so that a changed signature produces a recorded note
## rather than aborting the dump. A failure here is informative, not fatal.
## ---------------------------------------------------------------------------

hdr <- function(x) cat("\n\n", strrep("=", 72), "\n", x, "\n",
                       strrep("=", 72), "\n", sep = "")
sub <- function(x) cat("\n--- ", x, " ", strrep("-", max(0, 60 - nchar(x))),
                       "\n", sep = "")

show_api <- function(pkg, fn) {
  sub(paste0(pkg, "::", fn))
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat("  <", pkg, " not installed>\n", sep = ""); return(invisible())
  }
  obj <- tryCatch(get(fn, envir = asNamespace(pkg)), error = function(e) NULL)
  if (is.null(obj)) { cat("  <no such object>\n"); return(invisible()) }
  if (!is.function(obj)) { cat("  <", class(obj)[1], ">\n", sep = "");
                           return(invisible()) }
  print(args(obj))
}

probe <- function(label, expr) {
  sub(label)
  out <- tryCatch(expr, error = function(e) paste("ERROR:", conditionMessage(e)),
                  warning = function(w) paste("WARNING:", conditionMessage(w)))
  print(out)
  invisible(out)
}

## ---------------------------------------------------------------------------
hdr("VERSIONS")
pkgs <- c("sparseDist", "Matrix", "proxyC", "text2vec", "coop", "parallelDist",
          "philentropy", "BiocNeighbors", "bluster", "dbscan")
for (p in pkgs) {
  v <- tryCatch(as.character(utils::packageVersion(p)),
                error = function(e) "NOT INSTALLED")
  cat(sprintf("  %-16s %s\n", p, v))
}
cat("\n"); print(utils::sessionInfo()$R.version$version.string)
cat("BLAS: ", tryCatch(utils::sessionInfo()$BLAS, error = function(e) "?"), "\n")

## ---------------------------------------------------------------------------
hdr("SIGNATURES")

show_api("proxyC", "simil")
show_api("proxyC", "dist")

show_api("text2vec", "sim2")
show_api("text2vec", "dist2")

show_api("coop", "cosine")
show_api("coop", "pearson")
show_api("coop", "covar")
show_api("coop", "sparsity")

show_api("parallelDist", "parDist")
show_api("parallelDist", "parallelDist")

show_api("philentropy", "JSD")
show_api("philentropy", "distance")
show_api("philentropy", "jensen_shannon")

show_api("BiocNeighbors", "findKNN")
show_api("BiocNeighbors", "queryKNN")
show_api("BiocNeighbors", "ExhaustiveParam")
show_api("BiocNeighbors", "KmknnParam")
show_api("BiocNeighbors", "AnnoyParam")
show_api("BiocNeighbors", "HnswParam")

show_api("bluster", "neighborsToSNNGraph")
show_api("bluster", "makeSNNGraph")

show_api("dbscan", "kNN")
show_api("dbscan", "sNN")

## Ours, for side-by-side reference in the same dump.
show_api("sparseDist", "sparseDist")
show_api("sparseDist", "sparseKNN")
show_api("sparseDist", "sparseSNN")

## ---------------------------------------------------------------------------
hdr("DEFAULTS THAT CHANGE THE COMPARISON")

## Rounding. If proxyC rounds by default, every accuracy comparison must
## explicitly disable it -- otherwise we would report proxyC's rounding as
## sparseDist's numerical error.
probe("proxyC::simil formals: digits / min_simil / rank / use_nan",
      tryCatch({
        f <- formals(proxyC::simil)
        f[intersect(c("digits", "min_simil", "rank", "use_nan", "drop0",
                      "diag", "margin"), names(f))]
      }, error = function(e) conditionMessage(e)))

probe("proxyC::dist formals: digits / rank / margin",
      tryCatch({
        f <- formals(proxyC::dist)
        f[intersect(c("digits", "rank", "margin", "drop0", "diag"), names(f))]
      }, error = function(e) conditionMessage(e)))

probe("bluster::neighborsToSNNGraph formals",
      tryCatch(formals(bluster::neighborsToSNNGraph),
               error = function(e) conditionMessage(e)))

probe("BiocNeighbors::findKNN formals",
      tryCatch(formals(BiocNeighbors::findKNN),
               error = function(e) conditionMessage(e)))

## ---------------------------------------------------------------------------
hdr("BEHAVIOURAL PROBES")
## These answer questions no signature will: what the function actually
## computes. Small enough to verify by hand.

suppressMessages(library(Matrix))

## Two columns sharing a support pattern but with DIFFERENT values.
## sparseDist's Jaccard is defined on the sparsity pattern alone -- fastJacc
## counts iterator positions and never dereferences a value. If a competitor
## implements weighted/extended Jaccard, these two matrices give different
## answers and the comparison is between different functions, not an accuracy
## difference.
Xa <- as(as.matrix(data.frame(c1 = c(1, 1, 0, 0), c2 = c(1, 0, 1, 0))),
         "dgCMatrix")                       # binary
Xb <- as(as.matrix(data.frame(c1 = c(5, 3, 0, 0), c2 = c(2, 0, 7, 0))),
         "dgCMatrix")                       # same pattern, weighted

probe("sparseDist binary/Jaccard: binary vs weighted input (expect IDENTICAL)",
      tryCatch(list(
        binary   = sparseDist::sparseDist(Xa, method = "binary", full = TRUE,
                                          diag = TRUE, ncores = 1,
                                          verbose = FALSE),
        weighted = sparseDist::sparseDist(Xb, method = "binary", full = TRUE,
                                          diag = TRUE, ncores = 1,
                                          verbose = FALSE)),
        error = function(e) conditionMessage(e)))

probe("proxyC jaccard: binary vs weighted input (DIFFER => extended Jaccard)",
      tryCatch(list(
        binary   = as.matrix(proxyC::simil(Xa, method = "jaccard", margin = 2)),
        weighted = as.matrix(proxyC::simil(Xb, method = "jaccard", margin = 2))),
        error = function(e) conditionMessage(e)))

## text2vec is ROW-oriented, hence the transpose.
probe("text2vec jaccard: binary vs weighted input",
      tryCatch(list(
        binary   = as.matrix(text2vec::sim2(t(Xa), method = "jaccard")),
        weighted = as.matrix(text2vec::sim2(t(Xb), method = "jaccard"))),
        error = function(e) conditionMessage(e)))

## Degenerate columns. sparseDist defines J(empty,empty)=1, cos(empty,empty)=1,
## cos(empty,x)=0, undefined correlation -> 0. Competitors variously return
## NaN or 0. Max-abs-difference must be computed on a mask that excludes these,
## with the conventions reported separately -- one all-zero column would
## otherwise poison the headline accuracy number.
Xz <- as(as.matrix(data.frame(c1 = c(1, 1, 0), c2 = c(0, 0, 0),
                              c3 = c(0, 0, 0))), "dgCMatrix")

probe("sparseDist cosine with empty columns",
      tryCatch(sparseDist::sparseDist(Xz, method = "cosine", full = TRUE,
                                      diag = TRUE, ncores = 1, verbose = FALSE),
              error = function(e) conditionMessage(e)))
probe("proxyC cosine with empty columns",
      tryCatch(as.matrix(proxyC::simil(Xz, method = "cosine", margin = 2)),
               error = function(e) conditionMessage(e)))
probe("coop cosine with empty columns",
      tryCatch(coop::cosine(Xz), error = function(e) conditionMessage(e)))

## Jensen-Shannon. Ours is sqrt(JSD) in NATS, bounded by sqrt(log 2) ~ 0.8326.
## philentropy's JSD is a DIVERGENCE and defaults to log2. Comparing without
## fixing both the square root and the log base reports a constant factor as
## an accuracy failure.
P <- matrix(c(0.5, 0.3, 0.2,
              0.2, 0.3, 0.5,
              0.1, 0.1, 0.8), nrow = 3)          # columns sum to 1
probe("sparseDist js (expect sqrt of divergence, nats)",
      tryCatch(sparseDist::sparseDist(P, method = "js", full = TRUE,
                                      diag = TRUE, ncores = 1, verbose = FALSE),
               error = function(e) conditionMessage(e)))
probe("philentropy::JSD default (log2, divergence, rows are vectors)",
      tryCatch(philentropy::JSD(t(P)), error = function(e) conditionMessage(e)))
probe("philentropy::JSD unit='log' (nats, still a divergence)",
      tryCatch(philentropy::JSD(t(P), unit = "log"),
               error = function(e) conditionMessage(e)))

## SNN. Ours combines the two directions with pmax and includes each node in
## its own neighbour set by default. bluster is documented as also including
## self, but its edge combination and pruning rule need confirming, because
## edge-set agreement depends on both.
set.seed(1)
Xs <- abs(Matrix::rsparsematrix(50, 12, density = 0.4))
colnames(Xs) <- paste0("c", 1:12)
probe("sparseDist kNN indices (k=4)",
      tryCatch(sparseDist::sparseKNN(Xs, k = 4, method = "cosine", ncores = 1,
                                     verbose = FALSE)$idx,
               error = function(e) conditionMessage(e)))
probe("sparseSNN weights from those indices",
      tryCatch({
        nn <- sparseDist::sparseKNN(Xs, k = 4, method = "cosine", ncores = 1,
                                    verbose = FALSE)
        round(as.matrix(sparseDist::sparseSNN(nn, prune = 0, ncores = 1,
                                              verbose = FALSE)), 4)
      }, error = function(e) conditionMessage(e)))
probe("bluster::neighborsToSNNGraph on the same indices, type='jaccard'",
      tryCatch({
        nn <- sparseDist::sparseKNN(Xs, k = 4, method = "cosine", ncores = 1,
                                    verbose = FALSE)
        g  <- bluster::neighborsToSNNGraph(nn$idx, type = "jaccard")
        list(class = class(g), n_edges = igraph::ecount(g),
             weights = round(head(igraph::E(g)$weight, 20), 4))
      }, error = function(e) conditionMessage(e)))

hdr("END")
