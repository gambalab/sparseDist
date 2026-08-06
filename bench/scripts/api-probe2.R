#!/usr/bin/env Rscript
## ---------------------------------------------------------------------------
## api-probe2.R -- resolve the questions the first dump left open.
##
##   R_LIBS=$PWD/bench-lib Rscript bench/scripts/api-probe2.R \
##       > bench/results/api-probe2.txt 2>&1
##
## Each section answers one question that would otherwise be encoded into an
## adapter by guesswork. Everything is wrapped: a failure is a recorded result,
## not an abort.
## ---------------------------------------------------------------------------

hdr <- function(x) cat("\n\n", strrep("=", 72), "\n", x, "\n",
                       strrep("=", 72), "\n", sep = "")
sub <- function(x) cat("\n--- ", x, "\n", sep = "")
probe <- function(label, expr) {
  sub(label)
  print(tryCatch(expr,
                 error   = function(e) paste("ERROR:",   conditionMessage(e)),
                 warning = function(w) paste("WARNING:", conditionMessage(w))))
  invisible(NULL)
}
## Some probes need the VALUE despite a warning (proxyC warns on zero vectors
## but still returns a result), so warnings are suppressed rather than caught.
quiet <- function(expr) suppressWarnings(expr)

suppressMessages(library(Matrix))

Xa <- as(as.matrix(data.frame(c1 = c(1, 1, 0, 0), c2 = c(1, 0, 1, 0))),
         "dgCMatrix")
Xz <- as(as.matrix(data.frame(c1 = c(1, 1, 0), c2 = c(0, 0, 0),
                              c3 = c(0, 0, 0))), "dgCMatrix")
set.seed(42)
Xr <- abs(Matrix::rsparsematrix(200, 30, density = 0.25))
colnames(Xr) <- paste0("c", 1:30)

## ---------------------------------------------------------------------------
hdr("1. proxyC ROUNDING (digits = 14 is an ACTIVE default)")
## If digits cannot be raised, every accuracy comparison inherits proxyC's
## rounding and we would report it as sparseDist's numerical error.

probe("simil cosine, digits = 14 (default) vs digits = 22, max abs difference",
      quiet({
        a <- as.matrix(proxyC::simil(Xr, method = "cosine", margin = 2,
                                     digits = 14))
        b <- as.matrix(proxyC::simil(Xr, method = "cosine", margin = 2,
                                     digits = 22))
        c(max_abs_diff = max(abs(a - b)), any_different = !identical(a, b))
      }))

probe("does digits = Inf work?",
      quiet(max(as.matrix(proxyC::simil(Xr, method = "cosine", margin = 2,
                                        digits = Inf)))))

probe("sparseDist cosine SIMILARITY vs proxyC at digits = 22 (max abs diff)",
      quiet({
        ours <- 1 - sparseDist::sparseDist(Xr, method = "cosine", full = TRUE,
                                           diag = TRUE, dist = TRUE,
                                           ncores = 1, verbose = FALSE)
        theirs <- as.matrix(proxyC::simil(Xr, method = "cosine", margin = 2,
                                          digits = 22))
        max(abs(ours - theirs))
      }))

## ---------------------------------------------------------------------------
hdr("2. proxyC DEGENERATE COLUMNS (use_nan)")
## Ours: cos(0,0) = 1 similarity, cos(0,x) = 0. The first dump only captured
## proxyC's warning, not its values.

for (un in list(NULL, TRUE, FALSE)) {
  probe(paste0("proxyC cosine on all-zero columns, use_nan = ",
               if (is.null(un)) "NULL (default)" else un),
        quiet(as.matrix(proxyC::simil(Xz, method = "cosine", margin = 2,
                                      use_nan = un))))
}
probe("sparseDist cosine similarity on the same input (reference)",
      1 - sparseDist::sparseDist(Xz, method = "cosine", full = TRUE,
                                 diag = TRUE, dist = TRUE, ncores = 1,
                                 verbose = FALSE))
probe("proxyC jaccard on all-zero columns, use_nan = FALSE",
      quiet(as.matrix(proxyC::simil(Xz, method = "jaccard", margin = 2,
                                    use_nan = FALSE))))
probe("sparseDist binary distance on the same input (reference)",
      sparseDist::sparseDist(Xz, method = "binary", full = TRUE, diag = TRUE,
                             ncores = 1, verbose = FALSE))

## ---------------------------------------------------------------------------
hdr("3. proxyC 'jensen' -- an unplanned second JS comparator")
## philentropy is confirmed: ours == sqrt(JSD(unit = "log")). Whether proxyC's
## jensen is the divergence, its root, or a different log base is unknown.

P <- matrix(c(0.5, 0.3, 0.2, 0.2, 0.3, 0.5, 0.1, 0.1, 0.8), nrow = 3)
colnames(P) <- paste0("p", 1:3)
Ps <- as(P, "dgCMatrix")

probe("sparseDist js (reference)",
      sparseDist::sparseDist(P, method = "js", full = TRUE, diag = TRUE,
                             ncores = 1, verbose = FALSE))
probe("proxyC dist method = 'jensen', margin = 2",
      quiet(as.matrix(proxyC::dist(Ps, method = "jensen", margin = 2,
                                   digits = 22))))
probe("ratio and square-ratio vs ours (identifies sqrt / log base)",
      quiet({
        ours <- sparseDist::sparseDist(P, method = "js", full = TRUE,
                                       diag = TRUE, ncores = 1,
                                       verbose = FALSE)
        th <- as.matrix(proxyC::dist(Ps, method = "jensen", margin = 2,
                                     digits = 22))
        list(theirs_over_ours = th[1, 2] / ours[1, 2],
             theirs_over_ours_sq = th[1, 2] / (ours[1, 2]^2),
             sqrt_theirs_over_ours = sqrt(th[1, 2]) / ours[1, 2])
      }))

## ---------------------------------------------------------------------------
hdr("4. text2vec + Matrix 1.7.5 -- is the symmetric path broken?")
## sim2(x) failed coercing dsTMatrix -> dgCMatrix. If passing y explicitly
## avoids the symmetric return type, the comparator is salvageable.

probe("sim2(x) jaccard -- symmetric path",
      quiet(as.matrix(text2vec::sim2(t(Xa), method = "jaccard"))))
probe("sim2(x, x) jaccard -- explicit y",
      quiet(as.matrix(text2vec::sim2(t(Xa), t(Xa), method = "jaccard"))))
probe("sim2(x) cosine -- is cosine affected too?",
      quiet(as.matrix(text2vec::sim2(t(Xa), method = "cosine"))))
probe("sim2(x, x) cosine",
      quiet(as.matrix(text2vec::sim2(t(Xa), t(Xa), method = "cosine"))))
probe("dist2(x, x) jaccard",
      quiet(as.matrix(text2vec::dist2(t(Xa), t(Xa), method = "jaccard"))))
probe("class returned by sim2(x) cosine (unconverted)",
      quiet(class(text2vec::sim2(t(Xa), method = "cosine"))))

## ---------------------------------------------------------------------------
hdr("5. bluster SNN -- 31 edges (ours) vs 66 (theirs)")
## Weights match arithmetically (3/7, 1/9, 4/6 all reproduce), so the question
## is the EDGE SET, not the weighting. Counts cannot settle it; compare the
## adjacency matrices elementwise.

set.seed(1)
Xs <- abs(Matrix::rsparsematrix(50, 12, density = 0.4))
colnames(Xs) <- paste0("c", 1:12)

probe("elementwise comparison of full adjacency matrices",
      quiet({
        nn <- sparseDist::sparseKNN(Xs, k = 4, method = "cosine", ncores = 1,
                                    verbose = FALSE)
        ours <- as.matrix(sparseDist::sparseSNN(nn, prune = 0, ncores = 1,
                                                verbose = FALSE))
        g <- bluster::neighborsToSNNGraph(nn$idx, type = "jaccard")
        theirs <- as.matrix(igraph::as_adjacency_matrix(g, attr = "weight",
                                                        sparse = FALSE))
        dimnames(theirs) <- dimnames(ours)
        list(ours_nonzero          = sum(ours != 0) / 2,
             theirs_edges          = igraph::ecount(g),
             theirs_nonzero        = sum(theirs != 0) / 2,
             theirs_zero_weight_edges = sum(igraph::E(g)$weight == 0),
             max_abs_diff          = max(abs(ours - theirs)),
             n_disagreeing_cells   = sum(abs(ours - theirs) > 1e-10),
             ours_diag             = diag(ours),
             theirs_diag           = diag(theirs))
      }))

probe("first rows side by side (ours on top)",
      quiet({
        nn <- sparseDist::sparseKNN(Xs, k = 4, method = "cosine", ncores = 1,
                                    verbose = FALSE)
        ours <- as.matrix(sparseDist::sparseSNN(nn, prune = 0, ncores = 1,
                                                verbose = FALSE))
        g <- bluster::neighborsToSNNGraph(nn$idx, type = "jaccard")
        theirs <- as.matrix(igraph::as_adjacency_matrix(g, attr = "weight",
                                                        sparse = FALSE))
        round(rbind(ours = ours[1, ], theirs = theirs[1, ]), 4)
      }))

## sparseSNN defaults to symmetrise = TRUE (pmax over the two directions).
## If bluster does something else -- min, mean, or a directed graph -- the
## asymmetric case is where it shows.
probe("ours with symmetrise = FALSE, row 1 (is bluster using pmax?)",
      quiet({
        nn <- sparseDist::sparseKNN(Xs, k = 4, method = "cosine", ncores = 1,
                                    verbose = FALSE)
        round(as.matrix(sparseDist::sparseSNN(nn, prune = 0,
                                              symmetrise = FALSE, ncores = 1,
                                              verbose = FALSE))[1, ], 4)
      }))

## ---------------------------------------------------------------------------
hdr("6. BiocNeighbors -- does it densify sparse input?")
## Relevant to the MEMORY table, not just to speed: if findKNN coerces a
## dgCMatrix to dense, that cost belongs in the comparison and in the frontier.

probe("findKNN on a dgCMatrix (transposed: obs in rows)",
      quiet({
        r <- BiocNeighbors::findKNN(t(Xr), k = 5,
                                    BNPARAM = BiocNeighbors::ExhaustiveParam())
        list(names = names(r), idx_dim = dim(r$index),
             first_row = r$index[1, ])
      }))
probe("findKNN on a base dense matrix -- same answer?",
      quiet({
        a <- BiocNeighbors::findKNN(t(Xr), k = 5,
                                    BNPARAM = BiocNeighbors::ExhaustiveParam())
        b <- BiocNeighbors::findKNN(as.matrix(t(Xr)), k = 5,
                                    BNPARAM = BiocNeighbors::ExhaustiveParam())
        list(identical_indices = identical(a$index, b$index),
             max_dist_diff = max(abs(a$distance - b$distance)))
      }))
probe("ExhaustiveParam Cosine backend available?",
      quiet({
        r <- BiocNeighbors::findKNN(
          as.matrix(t(Xr)), k = 5,
          BNPARAM = BiocNeighbors::ExhaustiveParam(distance = "Cosine"))
        r$index[1, ]
      }))
probe("sparseDist cosine kNN on the same data (tie / ordering convention)",
      quiet(sparseDist::sparseKNN(Xr, k = 5, method = "cosine", ncores = 1,
                                  verbose = FALSE)$idx[1, ]))

## ---------------------------------------------------------------------------
hdr("7. coop -- pearson is named pcor; sparse support?")

probe("does coop::pcor exist, and with what signature?",
      tryCatch(args(coop::pcor), error = function(e) conditionMessage(e)))
probe("coop exported objects",
      tryCatch(ls(asNamespace("coop")), error = function(e) conditionMessage(e)))
probe("coop::cosine accepts dgCMatrix directly?",
      quiet({
        r <- coop::cosine(Xr)
        list(class = class(r), dim = dim(r), corner = round(r[1:3, 1:3], 6))
      }))
probe("coop::cosine vs sparseDist cosine similarity (max abs diff)",
      quiet({
        ours   <- 1 - sparseDist::sparseDist(Xr, method = "cosine",
                                             full = TRUE, diag = TRUE,
                                             ncores = 1, verbose = FALSE)
        theirs <- coop::cosine(Xr)
        max(abs(ours - theirs))
      }))

## ---------------------------------------------------------------------------
hdr("8. parallelDist -- dense baseline, methods and orientation")

probe("available methods",
      tryCatch(parallelDist::getDistMethods(),
               error = function(e) conditionMessage(e)))
probe("parDist euclidean vs sparseDist euclidean (max abs diff, obs in rows)",
      quiet({
        D <- as.matrix(parallelDist::parDist(as.matrix(t(Xr)),
                                             method = "euclidean",
                                             threads = 1))
        ours <- sparseDist::sparseDist(Xr, method = "euclidean", full = TRUE,
                                       diag = TRUE, ncores = 1,
                                       verbose = FALSE)
        dimnames(D) <- dimnames(ours)
        max(abs(D - ours))
      }))
probe("dist object size vs our dense matrix, same data",
      quiet({
        d <- parallelDist::parDist(as.matrix(t(Xr)), method = "euclidean",
                                   threads = 1)
        ours <- sparseDist::sparseDist(Xr, method = "euclidean", full = FALSE,
                                       diag = FALSE, ncores = 1,
                                       verbose = FALSE)
        c(parDist_mb = as.numeric(object.size(d)) / 1024^2,
          sparseDist_full_FALSE_mb = as.numeric(object.size(ours)) / 1024^2)
      }))

hdr("END")
