#' Optimized Distances calculation on sparse matrix
#'
#' Calculates distance matrices in parallel using multiple threads. Distances are computed among the cols of a sparse data matrix.
#' Code have been optimaized for highly sparse large matricies.
#'
#' @param X dgMatrix; a sparse matrix where columns are variables and rows observations.
#' @param Y dgMatrix; a sparse matrix where columns are variables and rows observations.
#' @param method string; a character string indicating which distance is to be computed. One of "binary" (i.e., jaccard) (default), "fJaccard", "euclidean","manhattan" or "pearson". Default binary
#' @param ncores numeric; number of cpu to use. Default value is 0 (i.e., number of cpus is auto detected and available number of cpu minus one are used).
#' @param verbose boolean; Show progress bar. Default is TRUE.
#' @param full boolean; Compute the full matrix (TRUE) or just low triangle (FALSE). Default is FALSE. This param id ignored when X and Y are used as input. If two matrices are passed as input the output matrix will be always a full matrix.
#' @param diag boolean; Compute diagonal. Default is FALSE.
#' @param dist boolean; If this parameter is set to FALSE, in the case of binary (i.e.,jaccard), fJaccard and pearson the similarity coefficient instead of the distance is returned . Defaullt value is TRUE and distance value is returned.
#'
#' @import Matrix
#' @import Rcpp
#' @export
sparseDist = function(X, Y=NULL,method="jaccard", ncores=0, verbose=T, full=F, diag=F, dist=T)
{
  if(ncores==0) {ncores=detectCores()}

  method = base::match.arg(arg = method,choices = c("binary","fJaccard","euclidean","manhattan","pearson","js","covariance"),several.ok = F)

  if (is.null(Y)){
    switch(method,
           binary = {r = fastJacc(m = X, ncores = ncores, verbose = verbose, full = full, diag = diag, dist = dist)
                      colnames(r) = rownames(r) = colnames(X)
                      return(r)
                    },
           fJaccard = {r = fastWeightedJacc(m = X, ncores = ncores, verbose = verbose, full = full, diag = diag, dist = dist)
                       colnames(r) = rownames(r) = colnames(X)
                       return(r)
                      },
           euclidean = {r = fastEuclidean(m = X, ncores = ncores, verbose = verbose, full = full, diag = diag)
                        colnames(r) = rownames(r) = colnames(X)
                      },
           manhattan = {r = fastManhattan(m = X, ncores = ncores, verbose = verbose, full = full, diag = diag)
                        colnames(r) = rownames(r) = colnames(X)
                        return(r)
                      },
           pearson = {r = fastCorr(m = X, ncores = ncores, verbose = verbose, full = full, diag = diag, dist = dist)
                      colnames(r) = rownames(r) = colnames(X)
                      return(r)
                    },
           js = {r = fastJS(m = as.matrix(X), ncores = ncores, verbose = verbose, full = full, diag = diag)
                 colnames(r) = rownames(r) = colnames(X)
                 return(r)
                },
           covariance = {r = fastCov(m = X, ncores = ncores, verbose = verbose, full = full, diag = diag)
                         colnames(r) = rownames(r) = colnames(X)
                         return(r)
                }
    )
  } else {
    switch(method,
           binary = {r = fastJacc2(m = X, m2 = Y, ncores = ncores, verbose = verbose, dist = dist)
                      colnames(r) = colnames(Y)
                      rownames(r) = colnames(X)
                      return(r)
                    },
           fJaccard = {r = fastWeightedJacc2(m = X, m2 = Y, ncores = ncores, verbose = verbose, dist = dist)
             colnames(r) = colnames(Y)
             rownames(r) = colnames(X)
             return(r)
             },
           euclidean = {r = fastEuclidean2(m = X, m2 = Y, ncores = ncores, verbose = verbose)
             colnames(r) = colnames(Y)
             rownames(r) = colnames(X)
             return(r)
           },
           manhattan = {r = fastManhattan2(m = X, m2 = Y, ncores = ncores, verbose = verbose)
             colnames(r) = colnames(Y)
             rownames(r) = colnames(X)
             return(r)
           },
           pearson ={r = fastCorr2(m = as.matrix(X), m2 = as.matrix(Y), ncores = ncores, verbose = verbose, dist = dist)
             colnames(r) = colnames(Y)
             rownames(r) = colnames(X)
             return(r)
           },
           js ={r = fastJS2(m = X, m2 = as.matrix(Y), ncores = ncores, verbose = verbose)
             colnames(r) = colnames(Y)
             rownames(r) = colnames(X)
             return(r)
           },
           covariance ={r = fastCov2(m = X, m2 = as.matrix(Y), ncores = ncores, verbose = verbose)
           colnames(r) = colnames(Y)
           rownames(r) = colnames(X)
           return(r)
           }
    )
  }
  return(NULL)
}

detectCores <- function() {
  ifelse(.Call("detectCoresCpp")>1,.Call("detectCoresCpp")-1,1)
}
