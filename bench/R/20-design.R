## ---------------------------------------------------------------------------
## 20-design.R -- which cells actually get run.
##
##   build_design(run_id) -> list of cell specs, ready for run_cell()
##
## A full cross of packages x methods x datasets x threads x phases x reps is
## roughly five thousand cells, most of them redundant. Instead there are five
## PANELS, each varying one axis and holding the rest fixed, so every figure
## answers one question and no cell is run to fill a table nobody reads.
##
## REPLICATION IS PER PANEL, not flat. Five reps of a four-hour frontier cell
## costs twenty hours to establish a yes/no. Feasibility gets 1, the size
## ladder 3, and the fast panels 5 (median + MAD as agreed).
## ---------------------------------------------------------------------------

## --- knobs ------------------------------------------------------------------

## 32 physical cores. 64 is reported as a SEPARATE hyperthreading footnote, not
## as another point on the same curve -- two hyperthreads are not two cores and
## plotting them together implies a scaling claim we cannot support.
BENCH_THREADS    <- c(1L, 2L, 4L, 8L, 16L, 32L)
BENCH_THREADS_HT <- 64L
BENCH_MAX_THREADS <- 32L

BENCH_REPS <- c(density = 5L, scaling = 5L, coercion = 5L,
                ladder = 3L, frontier = 1L, knn = 3L, snn = 3L)

BENCH_K <- 20L

## Approximate-kNN configurations, swept rather than taken at their defaults.
##
## An approximate method has an accuracy knob, and one point on that curve says
## almost nothing: tight settings are accurate and slow, loose ones fast and
## wrong. The ANN literature reports a recall-versus-time Pareto curve and a
## reviewer will expect one. Shared by the timing panel and the recall script
## so the two join on `variant`.
##
## The numbers are ef.search for HNSW and search.mult for Annoy.
## Dimensionality axis for the kNN panel, on real data.
##
## HNSW's recall is bounded by DIMENSIONALITY, not by ef.search: measured at
## 1.000 on a 50-component embedding and 0.969 on the same cells in 2000
## dimensions, with a tenfold increase in ef.search buying under two points.
## Graph descent stops discriminating once neighbour distances concentrate.
##
## This matters well beyond single-cell work, where PCA-first is a convention
## of the field rather than a property of the problem. Document-term matrices,
## chemical fingerprints, scATAC peaks and genomic feature matrices are all
## searched at thousands of dimensions with no projection step -- which is the
## regime sparseDist targets, and the regime where an exact method stays exact.
BENCH_DIM_FORMS <- c("pca10", "pca50", "pca200", "hvg")
BENCH_DIM_N     <- 10000L

BENCH_ANN_VARIANTS <- c("exact",
                        "annoy-10", "annoy-50", "annoy-200",
                        "hnsw-20", "hnsw-50", "hnsw-200")

## LADDER SIZES ARE BOUNDED BY THE REAL DATA.
##
## The 10x PBMC bundles hold 11769 cells (scRNA) and 8728 (scATAC), so any id
## above those returns the SAME matrix -- subsample_cols() simply hands back
## everything it has. Requesting 20k, 50k, 100k and 200k produced four
## identical points masquerading as a size sweep.
##
## Three genuinely distinct real sizes, capped below the scATAC count so the
## same ladder works for both assays.
BENCH_SIZES <- c(1000L, 5000L, 8000L)

## FRONTIER SIZES ARE SYNTHETIC, and deliberately so.
##
## The frontier asks where each method stops working, which is a property of
## the matrix DIMENSIONS and the memory budget -- not of what the columns mean.
## A dense p x p double result is 8p^2 bytes: 5 GB at 25k, 80 GB at 100k,
## 320 GB at 200k against a 234 GB node. Synthetic data tests that wall more
## cleanly than subsampling could, because density stays FIXED while p varies;
## a real subsample changes both at once.
##
## Density 0.01 is the realistic sparse regime and the one where the density
## panel says the merge walk pays.
BENCH_FRONTIER_SIZES <- c(25000L, 50000L, 100000L, 200000L)
BENCH_FRONTIER_ROWS  <- 2000L
BENCH_FRONTIER_DENS  <- 0.01

## Fixed reference dataset for the panels that vary something other than data.
BENCH_REF_DATASET <- "syn-n3000-p2000-d0.05-nonneg"

## --- helpers ----------------------------------------------------------------

timing_pairs <- function(experiment) {
  t <- ADAPTER_TABLE[ADAPTER_TABLE$experiment == experiment &
                     ADAPTER_TABLE$timing_ok, ]
  t[, c("package", "method")]
}

## Applicability needs dataset PROPERTIES (is it a simplex? is it signed?), and
## loading every dataset just to build the design would be absurd -- the 200k
## ones take minutes each. These are derivable from the id.
dataset_traits <- function(id) {
  spec <- parse_dataset_id(id)
  if (identical(spec$kind, "syn")) {
    return(list(simplex = isTRUE(spec$simplex), signed = spec$signed,
                binary = FALSE))
  }
  list(simplex = identical(spec$form, "simplex"),
       signed  = isTRUE(spec$is_pca),   # PCs are centred, so signed
       binary  = identical(spec$form, "bin"))
}

applicable_here <- function(package, method, id) {
  method_applicable(package, method, dataset_traits(id))
}

## --- panels -----------------------------------------------------------------

## 1. DENSITY SWEEP -- where does sparsity-aware traversal stop paying?
##
## Our per-pair merge walk costs O(p^2 * nnz-per-column); proxyC and text2vec
## compute a sparse matrix product and never touch a pair whose supports are
## disjoint, so their cost tracks nnz instead. There is a crossover, and at
## scRNA/scATAC sparsity it may not favour us. Finding it deliberately turns a
## potential weakness into a characterisation result.
panel_density <- function(run_id) {
  dens <- c(0.5, 0.1, 0.05, 0.01, 0.005, 0.001)
  pairs <- timing_pairs("pairwise")
  out <- list()

  emit <- function(id, keep) {
    for (i in which(keep)) {
      pk <- pairs$package[i]; mt <- pairs$method[i]
      if (!applicable_here(pk, mt, id)) next
      for (r in seq_len(BENCH_REPS[["density"]])) {
        out[[length(out) + 1L]] <<- new_cell_spec(
          run_id = run_id, panel = "density", experiment = "pairwise",
          package = pk, method = mt, dataset_id = id,
          threads = BENCH_MAX_THREADS, phase = "kernel", rep = r, seed = r)
      }
    }
  }

  ## Non-negative sweep: every method EXCEPT js, which needs a simplex.
  for (id in ids_density_sweep(n_cols = 3000L, n_rows = 2000L,
                               densities = dens)) {
    emit(id, pairs$method != "js")
  }

  ## Simplex sweep: js ONLY.
  ##
  ## Running the other methods here too would mostly re-measure the sweep
  ## above. Rescaling a column to sum 1 leaves cosine and Pearson EXACTLY
  ## unchanged -- both are invariant to per-column scaling -- and leaves the
  ## sparsity pattern, hence binary, untouched. Euclidean, manhattan and
  ## covariance do change, but only by a per-column constant that the
  ## non-negative sweep already characterises. That mistake cost 570
  ## duplicate cells in the first version of this design.
  for (id in ids_density_sweep_simplex(n_cols = 3000L, n_rows = 2000L,
                                       densities = dens)) {
    emit(id, pairs$method == "js")
  }
  out
}

## 2. SCALING -- does the OpenMP implementation actually scale?
##
## Only packages with real intra-op parallelism. coop is OpenMP, proxyC and
## parallelDist are TBB, and they are pinned through different variables --
## thread_env() sets all of them.
panel_scaling <- function(run_id) {
  pkgs <- c("sparseDist", "proxyC", "parallelDist", "coop")
  pairs <- timing_pairs("pairwise")
  pairs <- pairs[pairs$package %in% pkgs & pairs$method %in%
                 c("cosine", "binary"), ]
  id <- BENCH_REF_DATASET
  out <- list()
  for (th in c(BENCH_THREADS, BENCH_THREADS_HT)) {
    for (i in seq_len(nrow(pairs))) {
      pk <- pairs$package[i]; mt <- pairs$method[i]
      if (!applicable_here(pk, mt, id)) next
      for (r in seq_len(BENCH_REPS[["scaling"]])) {
        out[[length(out) + 1L]] <- new_cell_spec(
          run_id = run_id, panel = "scaling", experiment = "pairwise",
          package = pk, method = mt, dataset_id = id, threads = th,
          phase = "kernel", rep = r, seed = r,
          ## Tagged so the hyperthreaded point can be plotted apart from the
          ## physical-core curve without re-deriving it from the thread count.
          variant = if (th > BENCH_MAX_THREADS) "hyperthreaded" else NA)
      }
    }
  }
  out
}

## 3. COERCION -- what does matching each package's preferred layout cost?
##
## The only panel that runs both phases. t() on a dgCMatrix is a full CSC
## rebuild and text2vec needs one on every call; parallelDist needs a dense
## copy. Reporting kernel-only would flatter whichever package happens to share
## our layout, and reporting end-to-end only would understate the others'
## kernels. Both, side by side.
panel_coercion <- function(run_id) {
  pairs <- timing_pairs("pairwise")
  id <- BENCH_REF_DATASET
  out <- list()
  for (ph in c("kernel", "end_to_end")) {
    for (i in seq_len(nrow(pairs))) {
      pk <- pairs$package[i]; mt <- pairs$method[i]
      if (!applicable_here(pk, mt, id)) next
      for (r in seq_len(BENCH_REPS[["coercion"]])) {
        out[[length(out) + 1L]] <- new_cell_spec(
          run_id = run_id, panel = "coercion", experiment = "pairwise",
          package = pk, method = mt, dataset_id = id,
          threads = BENCH_MAX_THREADS, phase = ph, rep = r, seed = r)
      }
    }
  }
  out
}

## 4. SIZE LADDER and FRONTIER -- where does each approach stop working?
##
## Two panels over the same ids. "ladder" is the timing curve over sizes that
## complete; "frontier" is the feasibility question, one rep, run to sizes
## where the full-matrix methods are expected to die. A cell that returns
## status "killed" or "timeout" IS the datum there.
##
## Note the frontier is bounded by BOTH memory and wall clock: sparseKNN
## removes the O(p^2) memory wall but not the O(p^2) arithmetic, so at 200k it
## may hit the 4h cap with RAM to spare. Reporting the frontier as
## memory-only would invite exactly the question we cannot answer.
panel_ladder <- function(run_id) {
  ids <- ids_size_ladder("pbmc-rna-hvg", BENCH_SIZES)
  pairs <- timing_pairs("pairwise")
  out <- list()
  for (id in ids) {
    for (i in seq_len(nrow(pairs))) {
      pk <- pairs$package[i]; mt <- pairs$method[i]
      if (!applicable_here(pk, mt, id)) next
      for (r in seq_len(BENCH_REPS[["ladder"]])) {
        out[[length(out) + 1L]] <- new_cell_spec(
          run_id = run_id, panel = "ladder", experiment = "pairwise",
          package = pk, method = mt, dataset_id = id,
          threads = BENCH_MAX_THREADS, phase = "kernel", rep = r, seed = r)
      }
    }
  }
  out
}

## FRONTIER -- where does each method stop working?
##
## One rep: this is a yes/no, and a cell that fails is the datum. Rows with
## status "killed" or "timeout" must NOT be filtered out during analysis --
## they ARE the frontier.
##
## sparseKNN is included at every size because it is the point of the panel:
## it removes the O(p^2) MEMORY wall, though not the O(p^2) arithmetic, so it
## may still hit the wall-clock cap at the top end with RAM to spare. Reporting
## the frontier as memory-only would invite exactly the question we could not
## then answer, which is why the cap is stated alongside it.
panel_frontier <- function(run_id) {
  ids <- sprintf("syn-n%d-p%d-d%s-nonneg", BENCH_FRONTIER_SIZES,
                 BENCH_FRONTIER_ROWS,
                 format(BENCH_FRONTIER_DENS, scientific = FALSE, trim = TRUE))
  pairs <- timing_pairs("pairwise")
  out <- list()
  for (id in ids) {
    for (i in seq_len(nrow(pairs))) {
      pk <- pairs$package[i]; mt <- pairs$method[i]
      if (!applicable_here(pk, mt, id)) next
      out[[length(out) + 1L]] <- new_cell_spec(
        run_id = run_id, panel = "frontier", experiment = "frontier",
        package = pk, method = mt, dataset_id = id,
        threads = BENCH_MAX_THREADS, phase = "kernel", rep = 1L, seed = 1L)
    }
    ## The blocked kNN path, which is what the memory claim rests on.
    out[[length(out) + 1L]] <- new_cell_spec(
      run_id = run_id, panel = "frontier", experiment = "knn",
      package = "sparseDist", method = "binary", dataset_id = id,
      threads = BENCH_MAX_THREADS, phase = "kernel", rep = 1L, seed = 1L,
      k = BENCH_K, block_size = 256L)
  }
  out
}

## 5. kNN and SNN.
##
## Two SEPARATE kNN experiments, never merged: exact against exact, and exact
## against approximate with recall reported. Presenting an approximate method
## as simply faster is the single most common way a kNN benchmark misleads.
panel_knn <- function(run_id) {
  ## Both regimes: the sparse high-dimensional matrix (where exact search is
  ## the honest option) and the dense low-dimensional embedding (where
  ## BiocNeighbors is at home and approximate methods actually work). Reporting
  ## only the first would flatter us; only the second would miss the point.
  ## Both regimes: the sparse high-dimensional matrix (where exact search is
  ## the honest option) and the dense low-dimensional embedding (where
  ## BiocNeighbors is at home and approximate methods actually work).
  ids <- c(ids_size_ladder("pbmc-rna-hvg", BENCH_SIZES),
           ids_size_ladder("pbmc-rna-pca50", BENCH_SIZES))
  out <- list()
  for (id in ids) {
    for (mt in c("cosine", "euclidean")) {
      cells <- c(list(list(pkg = "sparseDist", variant = NA)),
                 lapply(BENCH_ANN_VARIANTS, function(v)
                   list(pkg = "BiocNeighbors", variant = v)))
      for (cc in cells) {
        for (r in seq_len(BENCH_REPS[["knn"]])) {
          out[[length(out) + 1L]] <- new_cell_spec(
            run_id = run_id, panel = "knn", experiment = "knn",
            package = cc$pkg, method = mt, dataset_id = id,
            threads = BENCH_MAX_THREADS, phase = "kernel", rep = r, seed = r,
            k = BENCH_K,
            block_size = if (identical(cc$pkg, "sparseDist")) 256L else NA,
            variant = cc$variant)
        }
      }
    }
  }
  out
}

panel_snn <- function(run_id) {
  ids <- c(ids_size_ladder("pbmc-rna-hvg", BENCH_SIZES),
           ids_size_ladder("pbmc-rna-pca50", BENCH_SIZES))
  out <- list()
  for (id in ids) {
    cells <- list(
      ## Verified identical to bluster (max abs difference 0), so this is an
      ## exact numerical comparison, not a structural one.
      list(pkg = "sparseDist", variant = NA),
      ## Our sparser default edge set: a structural contribution, compared
      ## against itself rather than against bluster.
      list(pkg = "sparseDist", variant = "edges=knn"),
      list(pkg = "bluster",    variant = NA))
    for (cc in cells) for (r in seq_len(BENCH_REPS[["snn"]])) {
      out[[length(out) + 1L]] <- new_cell_spec(
        run_id = run_id, panel = "snn", experiment = "snn",
        package = cc$pkg, method = "jaccard", dataset_id = id,
        threads = BENCH_MAX_THREADS, phase = "kernel", rep = r, seed = r,
        k = BENCH_K, variant = cc$variant)
    }
  }
  out
}

## 6. REAL DATA -- the assay-specific regimes the synthetic sweep cannot reach.
##
## The other panels run on log-normalised scRNA and on synthetic matrices, which
## between them miss three of the more interesting cases:
##
##   scATAC BINARY peaks  the sparsest regime in the benchmark, often under 1%,
##                        and the one method = "binary" was written for.
##   SIMPLEX profiles     the only valid input for Jensen-Shannon. Without it
##                        the js kernel and philentropy are never exercised at
##                        all -- the design as first written had zero js cells.
##   PCA EMBEDDING        dense, 50 dimensions, and the input Seurat and scanpy
##                        actually cluster on. It is parallelDist's natural
##                        home and the case where our CSC merge walk should
##                        LOSE to a contiguous dense loop. Reporting where the
##                        package should not be used is stronger than omitting
##                        it.
##
## Smaller size ladders than the main one: JS is dense and quadratic in
## features as well as observations, and these panels are about regime rather
## than scale.
panel_realdata <- function(run_id) {
  reps <- BENCH_REPS[["ladder"]]
  out <- list()
  emit <- function(id, pkg, method, variant = NA) {
    if (!applicable_here(pkg, method, id)) return(invisible())
    for (r in seq_len(reps)) {
      out[[length(out) + 1L]] <<- new_cell_spec(
        run_id = run_id, panel = "realdata", experiment = "pairwise",
        package = pkg, method = method, dataset_id = id,
        threads = BENCH_MAX_THREADS, phase = "kernel", rep = r, seed = r,
        variant = variant)
    }
  }

  ## scATAC, binary/Jaccard.
  for (n in c(1000L, 4000L, 8000L)) {   # scATAC holds 8728 cells
    id <- sprintf("pbmc-atac-bin-n%d", n)
    for (pk in c("sparseDist", "proxyC", "text2vec", "parallelDist")) {
      emit(id, pk, "binary")
    }
  }

  ## Jensen-Shannon. sparseDist against philentropy, which agree to seven
  ## significant figures (ours is sqrt of philentropy's JSD at unit = "log").
  ## proxyC was a third implementation until it was found to return an
  ## identically zero matrix on sparse simplex input -- see ADAPTER_TABLE.
  for (n in c(1000L, 5000L, 10000L)) {
    id <- sprintf("pbmc-rna-simplex-n%d", n)
    for (pk in c("sparseDist", "philentropy")) emit(id, pk, "js")
  }

  ## Dense embedding.
  for (n in c(1000L, 5000L, 10000L)) {
    id <- sprintf("pbmc-rna-pca50-n%d", n)
    for (mt in c("cosine", "euclidean")) {
      for (pk in c("sparseDist", "proxyC", "coop", "parallelDist",
                   "text2vec")) {
        if (!adapter_exists(pk, "pairwise", mt)) next
        emit(id, pk, mt)
      }
    }
  }
  out
}

## 7. DIMENSION -- recall and cost against ambient dimensionality.
##
## Fixed cell count, fixed k, varying only the number of dimensions: 10, 50 and
## 200 principal components against the full 2000-gene HVG matrix. Same cells
## throughout, so the only thing changing is the dimensionality of the space.
##
## The recall side comes from scripts/recall.R over the same ids; this panel
## supplies the matching TIMINGS, so the two join into a recall-versus-time
## curve per dimensionality.
panel_dimension <- function(run_id) {
  out <- list()
  for (form in BENCH_DIM_FORMS) {
    id <- sprintf("pbmc-rna-%s-n%d", form, BENCH_DIM_N)
    cells <- c(list(list(pkg = "sparseDist", variant = NA)),
               lapply(BENCH_ANN_VARIANTS, function(v)
                 list(pkg = "BiocNeighbors", variant = v)))
    for (cc in cells) for (r in seq_len(BENCH_REPS[["knn"]])) {
      out[[length(out) + 1L]] <- new_cell_spec(
        run_id = run_id, panel = "dimension", experiment = "knn",
        package = cc$pkg, method = "euclidean", dataset_id = id,
        threads = BENCH_MAX_THREADS, phase = "kernel", rep = r, seed = r,
        k = BENCH_K,
        block_size = if (identical(cc$pkg, "sparseDist")) 256L else NA,
        variant = cc$variant)
    }
  }
  out
}

## --- assembly ---------------------------------------------------------------

BENCH_PANELS <- c("density", "scaling", "coercion", "ladder", "frontier",
                  "realdata", "knn", "snn", "dimension")

build_design <- function(run_id, panels = BENCH_PANELS) {
  out <- list()
  add <- function(x) out <<- c(out, x)
  if ("ladder"   %in% panels) add(panel_ladder(run_id))
  if ("frontier" %in% panels) add(panel_frontier(run_id))
  if ("density"  %in% panels) add(panel_density(run_id))
  if ("scaling"  %in% panels) add(panel_scaling(run_id))
  if ("coercion" %in% panels) add(panel_coercion(run_id))
  if ("realdata" %in% panels) add(panel_realdata(run_id))
  if ("knn"      %in% panels) add(panel_knn(run_id))
  if ("snn"      %in% panels) add(panel_snn(run_id))
  if ("dimension" %in% panels) add(panel_dimension(run_id))

  ids <- vapply(out, function(s) s$cell_id, character(1))
  if (anyDuplicated(ids)) {
    stop("design contains duplicate cell_ids: ",
         paste(utils::head(unique(ids[duplicated(ids)]), 5), collapse = ", "),
         call. = FALSE)
  }
  out
}

design_summary <- function(design) {
  df <- data.frame(
    panel   = vapply(design, function(s) s$panel, character(1)),
    package = vapply(design, function(s) s$package, character(1)),
    threads = vapply(design, function(s) as.integer(s$threads), integer(1)),
    stringsAsFactors = FALSE)
  list(total = nrow(df),
       by_panel = as.data.frame(table(panel = df$panel)),
       by_package = as.data.frame(table(package = df$package)))
}
