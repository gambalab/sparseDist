#include <RcppArmadillo.h>
#include <progress.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif


using namespace std;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppProgress)]]

// Correlation coefficient between the columns of a dense matrix m
// [[Rcpp::export]]
arma::sp_mat fastCorr(const arma::sp_mat& m, int ncores=1, bool verbose=true, bool full=false, bool diag=true, bool dist=true)
{
  arma::mat d(m.n_cols,m.n_cols);
  d.zeros();
  int tot = (full) ? m.n_cols*m.n_cols - ((diag) ? m.n_cols : 0) : (m.n_cols*m.n_cols + ((diag) ? m.n_cols : 0))/2;
  Progress p(tot, verbose);
#pragma omp parallel for num_threads(ncores) shared(d)
  for(unsigned int i=0;i<m.n_cols;i++) {
    arma::vec li = arma::vec(m.col(i));
    for(unsigned int j = diag ? i : i+1;j<m.n_cols;j++) {
      if ( !Progress::check_abort())
      {
        p.increment(); // update progress
        d(i,j) = (dist) ? 1-arma::as_scalar(arma::cor(li,arma::vec(m.col(j)),0)) : arma::as_scalar(arma::cor(li,arma::vec(m.col(j)),0));
        d(j,i) = d(i,j);
      }
    }
  }
  return(arma::sp_mat(d));
}

// [[Rcpp::export]]
arma::sp_mat fastCorr2(const arma::sp_mat& m, const arma::mat& m2, int ncores=1, bool verbose=true, bool dist=true)
{

  if(m.n_rows != m2.n_rows) {
    Rcpp::stop("Mismatched row dimensions of m (%s) and m2 (%s).",m.n_rows, m2.n_rows);
  }

  arma::mat d(m.n_cols,m2.n_cols);
  d.zeros();

  unsigned int tot = m.n_cols*m2.n_cols;
  Progress p(tot, verbose);

#pragma omp parallel for num_threads(ncores) shared(d) collapse(2)
  for(unsigned int i=0;i<m.n_cols;i++) {
    for(unsigned int j=0;j<m2.n_cols;j++) {
      if ( !Progress::check_abort())
      {
        p.increment(); // update progress
        d(i,j) = (dist) ? 1-arma::as_scalar(arma::cor(arma::vec(m.col(i)),arma::vec(m2.col(j)))) : arma::as_scalar(arma::cor(arma::vec(m.col(i)),arma::vec(m2.col(j))));
      }
    }
  }
  return(arma::sp_mat(d));
}

// Correlation coefficient between the columns of a dense matrix m
// [[Rcpp::export]]
arma::sp_mat fastCov(const arma::sp_mat& m, int ncores=1, bool verbose=true, bool full=false, bool diag=true, bool dist=true)
{
  arma::mat d(m.n_cols,m.n_cols);
  d.zeros();
  int tot = (full) ? m.n_cols*m.n_cols - ((diag) ? m.n_cols : 0) : (m.n_cols*m.n_cols + ((diag) ? m.n_cols : 0))/2;
  Progress p(tot, verbose);
#pragma omp parallel for num_threads(ncores) shared(d)
  for(unsigned int i=0;i<m.n_cols;i++) {
    arma::vec li = arma::vec(m.col(i));
    for(unsigned int j = diag ? i : i+1;j<m.n_cols;j++) {
      if ( !Progress::check_abort())
      {
        p.increment(); // update progress
        d(i,j) = arma::as_scalar(arma::cov(li,arma::vec(m.col(j))));
        d(j,i) = d(i,j);
      }
    }
  }
  return(arma::sp_mat(d));
}

// [[Rcpp::export]]
arma::sp_mat fastCov2(const arma::sp_mat& m, const arma::mat& m2, int ncores=1, bool verbose=true)
{

  if(m.n_rows != m2.n_rows) {
    Rcpp::stop("Mismatched row dimensions of m (%s) and m2 (%s).",m.n_rows, m2.n_rows);
  }

  arma::mat d(m.n_cols,m2.n_cols);
  d.zeros();

  unsigned int tot = m.n_cols*m2.n_cols;
  Progress p(tot, verbose);

#pragma omp parallel for num_threads(ncores) shared(d) collapse(2)
  for(unsigned int i=0;i<m.n_cols;i++) {
    for(unsigned int j=0;j<m2.n_cols;j++) {
      if ( !Progress::check_abort())
      {
        p.increment(); // update progress
        d(i,j) = arma::as_scalar(arma::cov(arma::vec(m.col(i)),arma::vec(m2.col(j))));
      }
    }
  }
  return(arma::sp_mat(d));
}


// [[Rcpp::export]]
arma::sp_mat fastJacc2(const arma::sp_mat& m, const arma::sp_mat& m2, int ncores=1, bool verbose=true, bool dist=true)
{
  if(m.n_rows != m2.n_rows) {
    Rcpp::stop("Mismatched row dimensions of m (%s) and m2 (%s).",m.n_rows, m2.n_rows);
  }

  typedef arma::sp_mat::const_col_iterator iter;
  arma::mat d(m.n_cols,m2.n_cols);
  unsigned int tot = m.n_cols*m2.n_cols;

  Progress p(tot, verbose);
#pragma omp parallel for num_threads(ncores) shared(d) collapse(2)
  for(unsigned int i=0;i<m.n_cols;i++) {
    for(unsigned int j=0;j<m2.n_cols;j++) {
      if ( !Progress::check_abort())
      {
        p.increment(); // update progress
        iter i_iter = m.begin_col(i);
        iter j_iter = m2.begin_col(j);
        double common=0,j_count=0,i_count=0;

        while( (i_iter != m.end_col(i)) && (j_iter != m2.end_col(j)) )
        {
          if(i_iter.row() == j_iter.row())
          {
            i_count++;
            j_count++;
            common++;
            ++i_iter;
            ++j_iter;
          } else {
            if(i_iter.row() < j_iter.row())
            {
              i_count++;
              ++i_iter;
            } else {
              j_count++;
              ++j_iter;
            }
          }
        }
        for(; i_iter != m.end_col(i); ++i_iter) { i_count++; }
        for(; j_iter != m2.end_col(j); ++j_iter){ j_count++; }
        d(i,j) = (dist) ? 1 - (common / (i_count + j_count - common)) : common / (i_count + j_count - common);
      }
    }
  }

  return(arma::sp_mat(d));
}

// [[Rcpp::export]]
arma::sp_mat fastJacc(const arma::sp_mat& m, int ncores=1, bool verbose=true, bool full=false, bool diag=true, bool dist=true)
{
  typedef arma::sp_mat::const_col_iterator iter;
  arma::mat d(m.n_cols,m.n_cols);
  int tot = (full) ? m.n_cols*m.n_cols - ((diag) ? m.n_cols : 0) : (m.n_cols*m.n_cols + ((diag) ? m.n_cols : 0))/2;
  Progress p(tot, verbose);
#pragma omp parallel for num_threads(ncores) shared(d)
  for(unsigned int i=0;i<m.n_cols;i++) {
    for(unsigned int j = diag ? i : i+1;j<m.n_cols;j++) {
      if ( !Progress::check_abort())
      {
        p.increment(); // update progress
        iter i_iter = m.begin_col(i);
        iter j_iter = m.begin_col(j);
        double common=0,j_count=0,i_count=0;

        while( (i_iter != m.end_col(i)) && (j_iter != m.end_col(j)) )
        {
          if(i_iter.row() == j_iter.row())
          {
            i_count++;
            j_count++;
            common++;
            ++i_iter;
            ++j_iter;
          } else {
            if(i_iter.row() < j_iter.row())
            {
              i_count++;
              ++i_iter;
            } else {
              j_count++;
              ++j_iter;
            }
          }
        }
        for(; i_iter != m.end_col(i); ++i_iter) { i_count++; }
        for(; j_iter != m.end_col(j); ++j_iter){ j_count++; }

        d(j,i) = d(j,i) = (dist) ? 1 - (common / (i_count + j_count - common)) : common / (i_count + j_count - common);
        if (full) {d(i,j) = d(j,i);}
      }
    }
  }

  return(arma::sp_mat(d));
}

// [[Rcpp::export]]
arma::sp_mat fastWeightedJacc2(const arma::sp_mat& m, const arma::sp_mat& m2, int ncores=1, bool verbose=true, bool dist=true)
{
  if(m.n_rows != m2.n_rows) {
    Rcpp::stop("Mismatched row dimensions of m (%s) and m2 (%s).",m.n_rows, m2.n_rows);
  }

  typedef arma::sp_mat::const_col_iterator iter;
  arma::mat d(m.n_cols,m2.n_cols);
  unsigned int tot = m.n_cols*m2.n_cols;

  Progress p(tot, verbose);
#pragma omp parallel for num_threads(ncores) shared(d) collapse(2)
  for(unsigned int i=0;i<m.n_cols;i++) {
    for(unsigned int j=0;j<m2.n_cols;j++) {
      if ( !Progress::check_abort())
      {
        p.increment(); // update progress
        iter i_iter = m.begin_col(i);
        iter j_iter = m2.begin_col(j);

        double num=0,dem=0;
        int common=0;

        while( (i_iter != m.end_col(i)) && (j_iter != m2.end_col(j)) )
        {
          if(i_iter.row() == j_iter.row())
          {
            num+= std::min((*i_iter),(*j_iter));
            dem+= std::max((*i_iter),(*j_iter));
            common++;
            ++i_iter;
            ++j_iter;
          } else {
            if(i_iter.row() < j_iter.row())
            {
              dem+=(*i_iter);
              ++i_iter;
            } else {
              dem+=(*j_iter);
              ++j_iter;
            }
          }
        }
        for(; i_iter != m.end_col(i); ++i_iter) { dem+=(*i_iter); }
        for(; j_iter != m2.end_col(j); ++j_iter){ dem+=(*j_iter); }
        d(i,j) = (dist) ? 1 - (num / dem) : num/dem;
      }
    }
  }

  return(arma::sp_mat(d));
}

// [[Rcpp::export]]
arma::sp_mat fastWeightedJacc(const arma::sp_mat& m, int ncores=1,bool verbose=true, bool full=false, bool diag=true, bool dist=true)
{
  typedef arma::sp_mat::const_col_iterator iter;
  arma::mat d(m.n_cols,m.n_cols);
  int tot = (full) ? m.n_cols*m.n_cols - ((diag) ? m.n_cols : 0) : (m.n_cols*m.n_cols + ((diag) ? m.n_cols : 0))/2;
  Progress p(tot, verbose);
#pragma omp parallel for num_threads(ncores) shared(d)
  for(unsigned int i=0;i<m.n_cols;i++) {
    for(unsigned int j = diag ? i : i+1;j<m.n_cols;j++) {
      if ( !Progress::check_abort())
      {
        p.increment(); // update progress
        iter i_iter = m.begin_col(i);
        iter j_iter = m.begin_col(j);
        double num=0,dem=0;

        while( (i_iter != m.end_col(i)) && (j_iter != m.end_col(j)) )
        {
          if(i_iter.row() == j_iter.row())
          {
            num+= std::min((*i_iter),(*j_iter));
            dem+= std::max((*i_iter),(*j_iter));
            ++i_iter;
            ++j_iter;
          } else {
            if(i_iter.row() < j_iter.row())
            {
              dem+=(*i_iter);
              ++i_iter;
            } else {
              dem+=(*j_iter);
              ++j_iter;
            }
          }
        }
        for(; i_iter != m.end_col(i); ++i_iter) { dem+=(*i_iter); }
        for(; j_iter != m.end_col(j); ++j_iter){ dem+=(*j_iter); }
        d(j,i) = (dist) ? 1 - (num / dem) : num/dem;
        if (full) {d(i,j) = d(j,i);}
      }
    }
  }

  return(arma::sp_mat(d));
}

// [[Rcpp::export]]
arma::sp_mat fastManhattan(const arma::sp_mat& m, int ncores=1,bool verbose=true, bool full=false, bool diag=true)
{
  typedef arma::sp_mat::const_col_iterator iter;
  arma::mat d(m.n_cols,m.n_cols);
  int tot = (full) ? m.n_cols*m.n_cols - ((diag) ? m.n_cols : 0) : (m.n_cols*m.n_cols + ((diag) ? m.n_cols : 0))/2;
  Progress p(tot, verbose);
#pragma omp parallel for num_threads(ncores) shared(d)
  for(unsigned int i=0;i<m.n_cols;i++) {
    for(unsigned int j = diag ? i : i+1;j<m.n_cols;j++) {
      if ( !Progress::check_abort())
      {
        p.increment(); // update progress
        iter i_iter = m.begin_col(i);
        iter j_iter = m.begin_col(j);
        double num=0;

        while( (i_iter != m.end_col(i)) && (j_iter != m.end_col(j)) )
        {
          if(i_iter.row() == j_iter.row())
          {
            num+= std::abs((*i_iter)-(*j_iter));
            ++i_iter;
            ++j_iter;
          } else {
            if(i_iter.row() < j_iter.row())
            {
              num+=std::abs(*i_iter);
              ++i_iter;
            } else {
              num+=std::abs(*j_iter);
              ++j_iter;
            }
          }
        }
        for(; i_iter != m.end_col(i); ++i_iter) { num+=std::abs(*i_iter); }
        for(; j_iter != m.end_col(j); ++j_iter){ num+=std::abs(*j_iter); }
        d(j,i) = num;
        if (full) {d(i,j) = d(j,i);}
      }
    }
  }

  return(arma::sp_mat(d));
}

// [[Rcpp::export]]
arma::sp_mat fastManhattan2(const arma::sp_mat& m, const arma::sp_mat& m2, int ncores=1, bool verbose=true, bool dist=true)
{
  if(m.n_rows != m2.n_rows) {
    Rcpp::stop("Mismatched row dimensions of m (%s) and m2 (%s).",m.n_rows, m2.n_rows);
  }

  typedef arma::sp_mat::const_col_iterator iter;
  arma::mat d(m.n_cols,m2.n_cols);
  unsigned int tot = m.n_cols*m2.n_cols;

  Progress p(tot, verbose);
#pragma omp parallel for num_threads(ncores) shared(d) collapse(2)
  for(unsigned int i=0;i<m.n_cols;i++) {
    for(unsigned int j=0;j<m2.n_cols;j++) {
      if ( !Progress::check_abort())
      {
        p.increment(); // update progress
        iter i_iter = m.begin_col(i);
        iter j_iter = m2.begin_col(j);
        double num=0;

        while( (i_iter != m.end_col(i)) && (j_iter != m2.end_col(j)) )
        {
          if(i_iter.row() == j_iter.row())
          {
            num+= std::abs((*i_iter)-(*j_iter));
            ++i_iter;
            ++j_iter;
          } else {
            if(i_iter.row() < j_iter.row())
            {
              num+=std::abs(*i_iter);
              ++i_iter;
            } else {
              num+=std::abs(*j_iter);
              ++j_iter;
            }
          }
        }
        for(; i_iter != m.end_col(i); ++i_iter) { num+=std::abs(*i_iter); }
        for(; j_iter != m2.end_col(j); ++j_iter){ num+=std::abs(*j_iter); }
        d(i,j) = num;
      }
    }
  }

  return(arma::sp_mat(d));
}

// [[Rcpp::export]]
arma::sp_mat fastEuclidean(const arma::sp_mat& m, int ncores=1,bool verbose=true, bool full=false, bool diag=true)
{
  typedef arma::sp_mat::const_col_iterator iter;
  arma::mat d(m.n_cols,m.n_cols);
  int tot = (full) ? m.n_cols*m.n_cols - ((diag) ? m.n_cols : 0) : (m.n_cols*m.n_cols + ((diag) ? m.n_cols : 0))/2;
  Progress p(tot, verbose);
#pragma omp parallel for num_threads(ncores) shared(d)
  for(unsigned int i=0;i<m.n_cols;i++) {
    for(unsigned int j = diag ? i : i+1;j<m.n_cols;j++) {
      if ( !Progress::check_abort())
      {
        p.increment(); // update progress
        iter i_iter = m.begin_col(i);
        iter j_iter = m.begin_col(j);
        double num=0;

        while( (i_iter != m.end_col(i)) && (j_iter != m.end_col(j)) )
        {
          if(i_iter.row() == j_iter.row())
          {
            num+= (*i_iter - *j_iter)*(*i_iter - *j_iter);
            ++i_iter;
            ++j_iter;
          } else {
            if(i_iter.row() < j_iter.row())
            {
              num+=(*i_iter)*(*i_iter);
              ++i_iter;
            } else {
              num+=(*j_iter)*(*j_iter);
              ++j_iter;
            }
          }
        }
        for(; i_iter != m.end_col(i); ++i_iter) { num+=(*i_iter)*(*i_iter); }
        for(; j_iter != m.end_col(j); ++j_iter){ num+=(*j_iter)*(*j_iter); }
        d(j,i) = std::sqrt(num);
        if (full) {d(i,j) = d(j,i);}
      }
    }
  }

  return(arma::sp_mat(d));
}

// [[Rcpp::export]]
arma::sp_mat fastEuclidean2(const arma::sp_mat& m, const arma::sp_mat& m2, int ncores=1, bool verbose=true)
{
  if(m.n_rows != m2.n_rows) {
    Rcpp::stop("Mismatched row dimensions of m (%s) and m2 (%s).",m.n_rows, m2.n_rows);
  }

  typedef arma::sp_mat::const_col_iterator iter;
  arma::mat d(m.n_cols,m2.n_cols);
  unsigned int tot = m.n_cols*m2.n_cols;

  Progress p(tot, verbose);
#pragma omp parallel for num_threads(ncores) shared(d) collapse(2)
  for(unsigned int i=0;i<m.n_cols;i++) {
    for(unsigned int j=0;j<m2.n_cols;j++) {
      if ( !Progress::check_abort())
      {
        p.increment(); // update progress
        iter i_iter = m.begin_col(i);
        iter j_iter = m2.begin_col(j);
        double num=0;

        while( (i_iter != m.end_col(i)) && (j_iter != m2.end_col(j)) )
        {
          if(i_iter.row() == j_iter.row())
          {
            num+= (*i_iter - *j_iter)*(*i_iter - *j_iter);
            ++i_iter;
            ++j_iter;
          } else {
            if(i_iter.row() < j_iter.row())
            {
              num+=(*i_iter)*(*i_iter);
              ++i_iter;
            } else {
              num+=(*j_iter)*(*j_iter);
              ++j_iter;
            }
          }
        }
        for(; i_iter != m.end_col(i); ++i_iter) { num+=(*i_iter)*(*i_iter); }
        for(; j_iter != m2.end_col(j); ++j_iter){ num+=(*j_iter)*(*j_iter); }
        d(i,j) = std::sqrt(num);
      }
    }
  }

  return(arma::sp_mat(d));
}

// JS distance metric (sqrt(JS div)) between the columns of a dense matrix m
// returns vectorized version of the upper triangle (as R dist oject)
// [[Rcpp::export]]
arma::sp_mat fastJS(const arma::mat& m, int ncores=1,bool verbose=true, bool full=false, bool diag=true) {
  arma::mat d(m.n_cols,m.n_cols);
  int tot = (full) ? m.n_cols*m.n_cols - ((diag) ? m.n_cols : 0) : (m.n_cols*m.n_cols + ((diag) ? m.n_cols : 0))/2;
  Progress p(tot, verbose);
#pragma omp parallel for num_threads(ncores) shared(d)
  for(unsigned int i=0;i<m.n_cols;i++) {
    arma::vec li=log(m.col(i));
    for(unsigned int j = diag ? i : i+1;j<m.n_cols;j++) {
      if ( !Progress::check_abort())
      {
        p.increment(); // update progress
        arma::vec lj=log(m.col(j));
        arma::vec ji=m.col(j)+m.col(i);
        ji=m.col(j)%lj + m.col(i)%li - ji%(log(ji)-log(2.0));
        double v=arma::accu(ji.elem(arma::find_finite(ji)));
        if(v!=0) {
          d(j,i)=sqrt(v);
          if(full){d(i,j)=d(j,i);}
        }
      }
    }
  }

  return(arma::sp_mat(d));
}

// [[Rcpp::export]]
arma::sp_mat fastJS2(const arma::mat& m, const arma::mat& m2, int ncores=1, bool verbose=true)
{

  if(m.n_rows != m2.n_rows) {
    Rcpp::stop("Mismatched row dimensions of m (%s) and m2 (%s).",m.n_rows, m2.n_rows);
  }

  arma::mat d(m.n_cols,m2.n_cols);
  d.zeros();

  unsigned int tot = m.n_cols*m2.n_cols;
  Progress p(tot, verbose);

#pragma omp parallel for num_threads(ncores) shared(d) collapse(2)
  for(unsigned int i=0;i<m.n_cols;i++) {
    for(unsigned int j=0;j<m2.n_cols;j++) {
      if ( !Progress::check_abort())
      {
        p.increment(); // update progress
        arma::vec ji=m2.col(j)+m.col(i);
        ji=m2.col(j)%log(m2.col(j)) + m.col(i)%log(m.col(i)) - ji%(log(ji)-log(2.0));
        double v=arma::accu(ji.elem(arma::find_finite(ji)));
        if(v!=0) {
          d(i,j)=sqrt(v);
        }
      }
    }
  }
  return(arma::sp_mat(d));
}
