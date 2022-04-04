
#include <algorithm>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SVD>
#include <iostream>
#include <memory>
#include <nifti1_io.h>
#include <numeric>
#include <random>
#include <vector>

// #include "stickygpm/constants.h"
#include "stickygpm/eigen_slicing.h"
#include "stickygpm/extra_distributions.h"
#include "stickygpm/knots.h"
#include "stickygpm/nifti_manipulation.h"
#include "stickygpm/stickygpm_regression_data.h"
#include "stickygpm/truncated_normal_distribution.h"
#include "stickygpm/utilities.h"
#include "stickygpm/vector_summation.h"




#ifndef _STICKYGPM_PROJECTED_GP_REGRESSION_
#define _STICKYGPM_PROJECTED_GP_REGRESSION_


namespace stickygpm {


  
  
  template< typename RealType >
  class projected_gp_regression {
  public:
    typedef RealType scalar_type;
    typedef typename
      Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>
      matrix_type;
    typedef matrix_type param_type;
    typedef typename
      Eigen::SparseMatrix<scalar_type, Eigen::RowMajor>
      sparse_matrix_type;
    typedef typename Eigen::BDCSVD<matrix_type> svd_type;
    typedef typename
      Eigen::Matrix<scalar_type, Eigen::Dynamic, 1>
      vector_type;

    
    class shared_data {
    public:
      shared_data() { ; }

      template< typename CovT >
      shared_data(
        const ::nifti_image* const mask,
	const CovT& cov,
	const int nknots = 2048,
	const int neighborhood = 2000
      );

      double log_det_Sigma() const;
      // scalar_type cov( const scalar_type d );
      
      const matrix_type& knots() const;
      const matrix_type& Sigma_star() const;
      const Eigen::LLT<matrix_type>& Sigma_llt() const;
      // const matrix_type& Sigma_star_inv() const;
      // const matrix_type& Bases() const;
      const sparse_matrix_type& Bases() const;

      // const CovT& cov() const;
      
    private:
      // CovT _Covf;
      double _log_det_Sigma;
      matrix_type _knot_locations;
      matrix_type _Sigma_star;
      Eigen::LLT<matrix_type> _llt_of_Sigma;
      // matrix_type _Sigma_star_inv;
      // matrix_type _W;
      sparse_matrix_type _W;
    };



    projected_gp_regression();
    
    projected_gp_regression(
      const std::shared_ptr<const shared_data> shared_data_ptr,
      const int ncovars
    );

    projected_gp_regression(
      const projected_gp_regression<RealType>& other
    );


    void update(
      const stickygpm::stickygpm_regression_data<RealType>& data,
      const vector_type& sigma_sq_inv,
      const std::vector<int>& subset
    );
    void sample_from_prior();
    void accept_proposal();


    double log_kernel(
      const stickygpm::stickygpm_regression_data<RealType>& data,
      const vector_type& sigma_sq_inv,
      const int i
    ) const;
    
    double log_likelihood(
      const stickygpm::stickygpm_regression_data<RealType>& data,
      const vector_type& sigma_sq_inv,
      const int i
    ) const;
    
    double log_prior() const;
    constexpr double marginal_log_likelihood(
      const stickygpm::stickygpm_regression_data<RealType>& data,
      const vector_type& sigma_sq_inv,
      const int i
    ) { return 1; };

    double proposal_distance( const matrix_type& other_beta_star ) const;
    double distance( const matrix_type& other_beta_star ) const;

    scalar_type predictive_mean() const;
    scalar_type predictive_variance() const;

    int nknots() const;

    const param_type& parameter() const;
    const param_type& proposed_parameter() const;
    
    vector_type parameter_vector() const;
    vector_type projected_parameter_vector() const;
    vector_type projected_beta( const int j ) const;
    vector_type residuals(
      const stickygpm::stickygpm_regression_data<RealType>& data,
      const int i
    ) const;

    // vector_type pointwise_likelihood(
    //   const stickygpm::stickygpm_regression_data<RealType>& data,
    //   const vector_type& sigma_sq_inv,
    //   const int i
    // ) const;
    

  private:

    static const double _log_2pi;
    std::shared_ptr<const shared_data> _p_data;

    matrix_type _beta_star;      // dim = #(Knots) x #(covariates)
    matrix_type _proposed_beta_star;
    vector_type _rgauss;         // Std Gaussian samples - dim(_beta_star)
    Eigen::LLT<matrix_type> _llt_Vinv;

    void _draw_gaussian();
    matrix_type _projected_beta(const matrix_type& locations) const;
    
  };
  // class projected_gp_regression


  
}
// namespace stickygpm



template< typename RealType >
const double stickygpm::projected_gp_regression<RealType>
::_log_2pi = 1.837877066409345339082;



template< typename RealType >
stickygpm::projected_gp_regression<RealType>
::projected_gp_regression() {
  _beta_star = matrix_type::Zero(1, 1);
  _proposed_beta_star = _beta_star;
  _rgauss = vector_type::Zero(1);
};





template< typename RealType >
stickygpm::projected_gp_regression<RealType>
::projected_gp_regression(
  const typename std::shared_ptr
    < const stickygpm::projected_gp_regression<RealType>
      ::shared_data >
    shared_data_ptr,
  const int ncovars
) {
  assert( shared_data_ptr &&
	  "projected_gp_regression: null shared data pointer" );
  _p_data = shared_data_ptr;
  _beta_star = matrix_type::Zero( _p_data->knots().rows(), ncovars );
  _proposed_beta_star = _beta_star;
  _rgauss = vector_type::Zero( _beta_star.rows() );
};



template< typename RealType >
stickygpm::projected_gp_regression<RealType>::projected_gp_regression(
  const stickygpm::projected_gp_regression<RealType>& other
) {
  _p_data = other._p_data;
  _beta_star = other._beta_star;
  _proposed_beta_star = _beta_star;
  _rgauss = vector_type::Zero( _beta_star.rows() );
};





template< typename RealType >
void stickygpm::projected_gp_regression<RealType>::_draw_gaussian() {
  std::normal_distribution<scalar_type> Gaussian(0, 1);
  for (int i = 0; i < _rgauss.size(); i++) {
    _rgauss.coeffRef(i) = Gaussian( stickygpm::rng() );
  }
};








template< typename RealType >
void stickygpm::projected_gp_regression<RealType>
::sample_from_prior() {
  _llt_Vinv = _p_data->Sigma_star().llt();
  // Update coefficient
  for (int j = 0; j < _beta_star.cols(); j++) {
    _draw_gaussian();
    _proposed_beta_star.col(j) =
      _llt_Vinv.solve( _llt_Vinv.matrixL() * _rgauss );
  }
};




template< typename RealType >
void stickygpm::projected_gp_regression<RealType>::update(
  const stickygpm::stickygpm_regression_data<RealType>& data,
  const stickygpm::projected_gp_regression<RealType>::
    vector_type& sigma_sq_inv,
  const std::vector<int>& subset
) {
  assert( data.Y().rows() == sigma_sq_inv.size() &&
	  "projected_gp_regression::update : "
	  "sigma dimension mismatch" );
  assert( data.Y().rows() == _p_data->Bases().rows() &&
	  "projected_gp_regression::update : "
	  "mismatch of bases with outcome" );
  assert( data.X().cols() == _beta_star.cols() &&
	  "projected_gp_regression::update : "
	  "parameter dimension mismatch" );
  if ( subset.empty() ) {
    sample_from_prior();
    return;
  }
  // const int M = _beta_star.rows();
  const int M = data.Y().rows();
  const int P = _beta_star.cols();
  const int N = subset.size();
  
  const Eigen::VectorXi subset_ =
    Eigen::Map<const Eigen::VectorXi>( subset.data(), N );
  const Eigen::VectorXi all_cols_ =
    Eigen::VectorXi::LinSpaced( P, 0, P - 1 );

  const matrix_type Xsub = 
    stickygpm::nullary_index( data.X(), subset_, all_cols_ );
  const svd_type xsvd( Xsub,
    Eigen::DecompositionOptions::ComputeThinU |
    Eigen::DecompositionOptions::ComputeThinV
  );

  const matrix_type WtSW = _p_data->Bases().adjoint() *
    sigma_sq_inv.asDiagonal() * _p_data->Bases();

  //
  matrix_type xi_star = _beta_star *
    ( xsvd.matrixV() * xsvd.singularValues().asDiagonal() );
  matrix_type resid_hat( M, N );
  vector_type Uj( N );
  vector_type mu_hat( xi_star.rows() );
  // ^^ All of these change size with subset
  //   - Could preallocate and call *.conservativeResize()
  //
  
  scalar_type Djinv2;

  // Randomize order of coefficient updates
  std::vector<int> update_order( xsvd.rank() );
  std::iota( update_order.begin(), update_order.end(), 0 );
  std::shuffle(update_order.begin(), update_order.end(),
	       stickygpm::rng());


  // Compute current residuals -> resid_hat for subset
  resid_hat = _p_data->Bases() * _beta_star * -Xsub.transpose();
  // ^^ = -mu_hat
  for (int i = 0; i < N; i++) {
    resid_hat.col(i) += data.Y().col( subset[i] );
  }

  
  // Update coefficients
  for (int j : update_order) {

    for (int i = 0; i < N; i++) {
      // remove effect of xi_star_j in residuals
      Uj.coeffRef(i) = xsvd.matrixU().coeffRef( i, j );
      resid_hat.col(i) += Uj.coeffRef( i ) *
	_p_data->Bases() * xi_star.col( j );
    }
    mu_hat = _p_data->Bases().adjoint() *
      ( sigma_sq_inv.asDiagonal() * (resid_hat * Uj) );

    // Update coefficient
    _draw_gaussian();
    Djinv2 = xsvd.singularValues().coeffRef( j );
    Djinv2 = 1 / ( Djinv2 * Djinv2 );
    _llt_Vinv = ( WtSW + Djinv2 * _p_data->Sigma_star() ).llt();
    xi_star.col(j) =
      _llt_Vinv.solve( mu_hat + _llt_Vinv.matrixL() * _rgauss );

    // Update residuals
    for (int i = 0; i < N; i++) {
      resid_hat.col(i) -= Uj.coeffRef( i ) *
	_p_data->Bases() * xi_star.col( j );
    }
  }

  // Put parameter back in native orientation
  _proposed_beta_star = xi_star *
    (xsvd.singularValues().cwiseInverse().asDiagonal() *
     xsvd.matrixV().transpose());
  
};





template< typename RealType >
void stickygpm::projected_gp_regression<RealType>::accept_proposal() {
  _beta_star = _proposed_beta_star;
};








template< typename RealType >
double stickygpm::projected_gp_regression<RealType>::log_kernel(
  const stickygpm::stickygpm_regression_data<RealType>& data,
  const typename
    stickygpm::projected_gp_regression<RealType>::vector_type&
    sigma_sq_inv,
  const int i
) const {
  assert( i >= 0 && i < data.n() &&
	  "projected_gp_regression::log_kernel : "
	  "subject index i outside range" );
  assert( data.Y().rows() == sigma_sq_inv.size() &&
	  "projected_gp_regression::log_kernel : "
	  "sigma dimension mismatch" );
  assert( data.Y().rows() == _p_data->Bases().rows() &&
	  "projected_gp_regression::log_kernel : "
	  "mismatch of bases with outcome" );
  assert( data.X().cols() == _beta_star.cols() &&
	  "projected_gp_regression::log_kernel : "
	  "parameter dimension mismatch" );
  const vector_type temp =
    sigma_sq_inv.cwiseSqrt().asDiagonal() *
    ( data.Y().col( i ) - _p_data->Bases() *
      ( _beta_star * data.X().row( i ).transpose() )
      );
  return -0.5 * stickygpm::vdot( temp );
};





template< typename RealType >
double stickygpm::projected_gp_regression<RealType>::log_likelihood(
  const stickygpm::stickygpm_regression_data<RealType>& data,
  const typename
    stickygpm::projected_gp_regression<RealType>::vector_type&
    sigma_sq_inv,
  const int i
) const {
  const double norm_c = -0.5 * sigma_sq_inv.size() * _log_2pi;
  double lp = log_kernel(data, sigma_sq_inv, i) + norm_c;
  for ( int j = 0; j < sigma_sq_inv.size(); j++ ) {
    lp += 0.5 *
      std::log( static_cast<double>(sigma_sq_inv.coeff(j)) );
  }
  return lp;
};








template< typename RealType >
double stickygpm::projected_gp_regression<RealType>
::log_prior() const {
  // const double lp = _p_data->Sigma_llt().solve(
  //   _p_data->Sigma_llt().matrixL() * _beta_star
  // ).template cast<double>().colwise().squaredNorm().sum();
  const double lp = (
    _p_data->Sigma_llt().matrixU() * _beta_star
  ).template cast<double>().colwise().squaredNorm().sum();
  // std::cout << "(kern = " << lp << ";  log|S_*| = "
  // 	    << _p_data->log_det_Sigma() << ")\n";
  const double norm_c = _p_data->log_det_Sigma() * _beta_star.cols() -
    _log_2pi * _beta_star.size();
  return -0.5 * lp + 0.5 * norm_c;
};





template< typename RealType >
double stickygpm::projected_gp_regression<RealType>
::proposal_distance( const matrix_type& other_beta_star ) const {
  const double d2 = (other_beta_star - _proposed_beta_star)
    .template cast<double>().colwise().squaredNorm().mean();
  return std::sqrt( d2 );
};


template< typename RealType >
double stickygpm::projected_gp_regression<RealType>
::distance( const matrix_type& other_beta_star ) const {
  const double d2 = (other_beta_star - _beta_star)
    .template cast<double>().colwise().squaredNorm().mean();
  return std::sqrt( d2 );
};






template< typename RealType >
RealType stickygpm::projected_gp_regression<RealType>
::predictive_mean() const {
  return -1;
};


template< typename RealType >
RealType stickygpm::projected_gp_regression<RealType>
::predictive_variance() const {
  return -1;
};




template< typename RealType >
int stickygpm::projected_gp_regression<RealType>::nknots() const {
  return _p_data->knots().rows();
};




template< typename RealType >
const typename stickygpm::projected_gp_regression
  <RealType>::param_type&
stickygpm::projected_gp_regression<RealType>::parameter() const {
  return _beta_star;
};




template< typename RealType >
const typename stickygpm::projected_gp_regression
  <RealType>::param_type&
stickygpm::projected_gp_regression<RealType>::proposed_parameter() const {
  return _proposed_beta_star;
};




template< typename RealType >
typename stickygpm::projected_gp_regression
  <RealType>::vector_type
stickygpm::projected_gp_regression<RealType>::parameter_vector() const {
  const int n = _beta_star.cols();
  const int p = _beta_star.rows();
  // vector_type b = Eigen::Map<vector_type>(_beta_star.data(), n * p, 1);
  // ^^ does not work if parameters() is marked const
  vector_type b( n * p );
  int bi = 0;
  for ( int j = 0; j < _beta_star.cols(); j++ ) {
    for ( int i = 0; i < _beta_star.rows(); i++ ) {
      b.coeffRef( bi ) = _beta_star.coeffRef( i, j );
      bi++;
    }
  }
  return b;
};




template< typename RealType >
typename stickygpm::projected_gp_regression
  <RealType>::vector_type
stickygpm::projected_gp_regression<RealType>::projected_parameter_vector() const {
  const int n = _p_data->Bases().rows();
  const int p = _beta_star.cols();
  // vector_type b = Eigen::Map<vector_type>(_beta_star.data(), n * p, 1);
  // ^^ does not work if parameters() is marked const
  param_type projb = _p_data->Bases() * _beta_star;
  vector_type b( Eigen::Map<vector_type>(projb.data(), n * p) );
  return b;
};







template< typename RealType >
typename stickygpm::projected_gp_regression
  <RealType>::vector_type
stickygpm::projected_gp_regression<RealType>::projected_beta(
  const int j
) const {
  assert( j >= 0 && j < _beta_star.cols() &&
	  "projected_gp_regression::projected_beta: bad index" );
  return _p_data->Bases() * _beta_star.col( j );
};




template< typename RealType >
typename stickygpm::projected_gp_regression
  <RealType>::vector_type
stickygpm::projected_gp_regression<RealType>::residuals(
  const stickygpm::stickygpm_regression_data<RealType>& data,
  const int i
) const {
  assert( i >= 0 && i < data.n() &&
	  "projected_gp_regression::residuals : "
	  "subject index i outside range" );
  return data.Y().col(i) - _p_data->Bases() *
    ( _beta_star * data.X().row(i).transpose() );
};





// template< typename RealType >
// typename stickygpm::projected_gp_regression
//   <RealType>::vector_type
// stickygpm::projected_gp_regression<RealType>::pointwise_likelihood(
//   const stickygpm::stickygpm_regression_data<RealType>& data,
//   const typename
//     stickygpm::projected_gp_regression<RealType>::vector_type&
//     sigma_sq_inv,
//   const int i
// ) const {
//   static const scalar_type norm_c = 1 / std::sqrt(2 * PI);
//   vector_type pl =
//     norm_c * sigma_sq_inv.cwiseSqrt().asDiagonal() *
//     ( static_cast<scalar_type>(-0.5) *
//       sigma_sq_inv.cwiseSqrt().asDiagonal() *
//       ( data.Y().col( i ) - _p_data->Bases() *
// 	( _beta_star * data.X().row(i).transpose() )
// 	)
//       ).array().exp().matrix();
//   return pl;
// }







// --- projected_gp_regression<RealType>::shared_data ----------------



template< typename RealType >
template< typename CovT >
stickygpm::projected_gp_regression<RealType>::shared_data
::shared_data(
  const ::nifti_image* const mask,
  const CovT& cov,
  const int nknots,
  const int neighborhood
) {
  const matrix_type xyz = stickygpm::get_nonzero_xyz(mask)
    .template cast<scalar_type>();
  // _Covf = cov;
  _knot_locations =
    stickygpm::get_knot_positions_uniform<scalar_type>(mask, nknots);
  _Sigma_star =
    stickygpm::knot_covariance_matrix( _knot_locations, cov );
  _llt_of_Sigma = _Sigma_star.llt();
  _log_det_Sigma = 0;
  for ( int i = 0; i < _Sigma_star.rows(); i++ ) {
    _log_det_Sigma += 2 * std::log(
      (double)_llt_of_Sigma.matrixL().coeff(i, i)
    );
  }
  // _Sigma_star_inv = _Sigma_star.llt()
  //   .solve( matrix_type::Identity(nknots, nknots) );
  // _W = stickygpm::basis_matrix(xyz, _knot_locations, cov);
  _W = stickygpm::knn_basis_matrix2(
    xyz, _knot_locations, cov, neighborhood );

  //
  std::cout << "\n\t(Minimax knot distance: "
	    << stickygpm::minimax_knot_distance(mask, _knot_locations)
	    << " mm)\n";
};


template< typename RealType >
const typename
stickygpm::projected_gp_regression<RealType>::matrix_type&
stickygpm::projected_gp_regression<RealType>::shared_data
::knots() const {
  return _knot_locations;
};


template< typename RealType >
const typename
stickygpm::projected_gp_regression<RealType>::matrix_type&
stickygpm::projected_gp_regression<RealType>::shared_data
::Sigma_star() const {
  return _Sigma_star;
};




template< typename RealType >
const Eigen::LLT<
  typename stickygpm::projected_gp_regression<RealType>::matrix_type
  >&
stickygpm::projected_gp_regression<RealType>::shared_data
::Sigma_llt() const {
  return _llt_of_Sigma;
};




template< typename RealType >
double stickygpm::projected_gp_regression<RealType>::shared_data
::log_det_Sigma() const {
  return _log_det_Sigma;
};


// template< typename RealType >
// const typename
// stickygpm::projected_gp_regression<RealType>::matrix_type&
// stickygpm::projected_gp_regression<RealType>::shared_data
// ::Sigma_star_inv() const {
//   return _Sigma_star_inv;
// };


template< typename RealType >
const typename
stickygpm::projected_gp_regression<RealType>::sparse_matrix_type&
stickygpm::projected_gp_regression<RealType>::shared_data
::Bases() const {
  return _W;
};



// template< typename RealType >
// RealType stickygpm::projected_gp_regression<RealType>::shared_data
// ::cov( const RealType d ) const {
//   return _Covf(d);
// };


// template< typename RealType >
// CovT stickygpm::projected_gp_regression<RealType>::shared_data
// ::cov() const {
//   return _Covf;
// };



#endif  // _STICKYGPM_PROJECTED_GP_REGRESSION_



