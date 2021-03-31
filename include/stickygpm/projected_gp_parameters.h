
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <iostream>
#include <nifti1_io.h>
#include <random>
#include <vector>

#include "stickygpm/extra_distributions.h"
#include "stickygpm/knots.h"
#include "stickygpm/truncated_normal_distribution.h"
#include "stickygpm/utilities.h"
#include "stickygpm/vector_summation.h"




#ifndef _STICKYGPM_PROJECTED_GP_PARAMETERS_
#define _STICKYGPM_PROJECTED_GP_PARAMETERS_


namespace stickygpm {

  template< typename RealType >
  class projected_gp_parameters {
  public:
    typedef RealType scalar_type;
    typedef typename
      Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>
      matrix_type;
    typedef typename
      Eigen::SparseMatrix<scalar_type, Eigen::RowMajor>
      sparse_matrix_type;
    typedef typename
      Eigen::Matrix<scalar_type, Eigen::Dynamic, 1>
      vector_type;


    projected_gp_parameters();
    
    projected_gp_parameters(
      const ::nifti_image* const mask,
      const std::vector<RealType>& theta,
      const int nknots = 2048
    );

    void update(
      const matrix_type& Yproj,
      const vector_type& sigma_sq_inv
    );

    void update(
      const matrix_type& Yproj,
      const vector_type& sigma_sq_inv,
      const std::vector<int>& subset
    );


    double log_likelihood(
      const matrix_type& Yproj,
      const vector_type& sigma_sq_inv,
      const int which
    ) const ;

    const sparse_matrix_type& xcor_matrix() const;
    // const vector_type& delta() const;
    vector_type mu() const;
    

  private:
    std::normal_distribution<scalar_type> _Gaussian;
    std::uniform_real_distribution<scalar_type> _Uniform;
    std::vector<scalar_type> _theta;

    matrix_type _Ks;
    matrix_type _Vmu_inv;
    sparse_matrix_type _XCor;

    Eigen::LLT<matrix_type> _llt_Vmu;
    Eigen::LLT<matrix_type> _lltGamma;

    // vector_type _delta;  // mixed use - dim(nvox)
    vector_type _eta;    // mixed use - dim(nvox)
    vector_type _gamma;  // latent field - sparsity - dim(nvox)
    vector_type _mu_k;   // dim = #(Knots)
    // vector_type __mu;    // dim(nvox)
    vector_type _xi;     // Std Gaussian samples - dim(_mu_k)

    void _compute_XCor(
      const ::nifti_image* const mask,
      const matrix_type& Knots,
      const std::vector<scalar_type>& theta
    );

    // Need to write strict prior sample update methods too
    void _sample_xi();
    // void _update_gamma();
    // void _update_gamma_delta(
    //   const matrix_type& Yproj,
    //   const vector_type&sigma_sq_inv
    // );
    void _update_mu(
      const matrix_type& Yproj,
      const vector_type& sigma_sq_inv
    );
    void _update_mu(
      const matrix_type& Yproj,
      const vector_type& sigma_sq_inv,
      const std::vector<int>& subset
    );
    
  };
  // class projected_gp_parameters


  
}
// namespace stickygpm








template< typename RealType >
stickygpm::projected_gp_parameters<RealType>::projected_gp_parameters(
) {
  const int nknots = 2048;
  _Gaussian = std::normal_distribution<scalar_type>(0, 1);
  _Uniform = std::uniform_real_distribution<scalar_type>(0, 1);
  _theta = std::vector<scalar_type>{1, 0.08, 1.999};
  _mu_k = vector_type::Zero(nknots);
  _xi = vector_type::Zero(nknots);
};





template< typename RealType >
stickygpm::projected_gp_parameters<RealType>::projected_gp_parameters(
  const ::nifti_image* const mask,
  const std::vector<RealType>& theta,
  const int nknots
) {
  _Gaussian = std::normal_distribution<scalar_type>(0, 1);
  _Uniform = std::uniform_real_distribution<scalar_type>(0, 1);
  _theta = std::vector<scalar_type>(theta);

//////////////////////////////////////////////////////////////////////
  matrix_type Knots =
    stickygpm::get_knot_positions_uniform<scalar_type>(mask, nknots);

  _Ks = stickygpm::projection_covariance_matrix(Knots, theta);
  _compute_XCor(mask, Knots, _theta);
  std::cout << "Sparse Cross-Correlation matrix has "
	    << ((int)_XCor.nonZeros())
	    << " non-zero elements ("
	    << (100.0 - (double)_XCor.nonZeros() /
		(_XCor.rows() * _XCor.cols()) * 100)
	    << "% sparse)"
	    << std::endl;
  const int nvox = _XCor.cols();

  // _delta = vector_type::Zero(nvox);
  // _gamma = vector_type::Zero(nvox);
  // __mu = vector_type::Zero(nvox);
  _eta = vector_type::Zero(nvox);
  
  _mu_k = vector_type::Zero(nknots);
  _xi = vector_type::Zero(nknots);

  matrix_type PtP = ( _XCor * _XCor.adjoint() );
  // std::cout << (PtP.template block<20,20>(0, 0))
  // 	    << std::endl << std::endl
  // 	    << (_K.template block<20,20>(0, 0))
  // 	    << std::endl
  // 	    << std::endl;
  _Vmu_inv = PtP + _Ks;
  _llt_Vmu = _Vmu_inv.llt();

  // PtP /= _theta[0];  // re-use of memory from PtP computation
  // PtP += _Ks;
  // PtP /= _theta[0];
  // _lltGamma = PtP.llt();
  
  // // Sample initial parameter values
  // _sample_xi();
  // _gamma = ( _lltGamma.solve(_lltGamma.matrixL() * _xi)
  // 	     .transpose() * _XCor ).transpose();
  // //
  // for (int i = 0; i < _delta.size(); i++) {
  //   if (_gamma.coeffRef(i) <= 0)
  //     _delta.coeffRef(i) = 0;
  // }
  //
  
  // _sample_xi();
  // _mu_k = _llt_Vmu.solve( _llt_Vmu.matrixL() * _xi );
  // __mu = ( _mu_k.transpose() * _XCor ).transpose();
  // __mu = vector_type::Zero(nvox);
};






template< typename RealType >
void stickygpm::projected_gp_parameters<RealType>::_compute_XCor(
  const ::nifti_image* const mask,
  const typename stickygpm::projected_gp_parameters
    <RealType>::matrix_type& Knots,
  const typename std::vector<stickygpm::projected_gp_parameters
    <RealType>::scalar_type>& theta
) {
  stickygpm::sparse_matrix_data<scalar_type> Pm =
    stickygpm::get_sparse_crosscorrelation_data
    <scalar_type>(mask, Knots, theta);

  _XCor = Eigen::Map<sparse_matrix_type>(
    Pm.nrow, Pm.ncol, Pm._Data.size(),
    Pm.cum_row_counts.data(), Pm.column_indices.data(),
    Pm._Data.data());
};



template< typename RealType >
void stickygpm::projected_gp_parameters<RealType>::_sample_xi() {
  for (int i = 0; i < _xi.size(); i++)
    _xi.coeffRef(i) = _Gaussian(stickygpm::rng());
};





// template< typename RealType >
// void stickygpm::projected_gp_parameters<RealType>::_update_gamma() {
//   _sample_xi();
//   _gamma =
//     ( _lltGamma.solve(_XCor * _delta / _theta[0] +
// 			   _lltGamma.matrixL() * _xi).transpose() *
//       _XCor ).transpose() / _theta[0];
// };




// template< typename RealType >
// void stickygpm::projected_gp_parameters
// <RealType>::_update_gamma_delta(
//   const typename
//     stickygpm::projected_gp_parameters<RealType>::matrix_type& Yproj,
//   const typename
//     stickygpm::projected_gp_parameters
//     <RealType>::vector_type&sigma_sq_inv
// ) {
//   // 1) Sample latent truncated normal field into _delta
//   // 2) Update _gamma | (Trunc-Norm field)
//   // 3) Sample binary field into _delta (update _delta | _gamma)
//   //

//   // 1)
//   scalar_type upper, lower;
//   for (int i = 0; i < _delta.size(); i++) {
//     if (_delta.coeffRef(i) == 1) {
//       upper = 1e4;
//       lower = 0;
//     }
//     else {
//       upper = 0;
//       lower = -1e4;
//     }
//     truncated_normal_distribution<scalar_type>
//       _TN(_gamma.coeffRef(i), 1.0, lower, upper);
//     _delta.coeffRef(i) = _TN(stickygpm::rng());
//   }
//   //

//   // 2)
//   _update_gamma();
//   //

//   // 3)
//   const double eps0 = 1e-6;
//   double logpr1, logpr0, Pr_delta_1;
//   double prior_pr1, unnorm_post_pr1, unnorm_post_pr0;
//   double mu0, var_mu0_inv;  // <- sufficient stats
//   for (int i = 0; i < _delta.size(); i++) {
//     mu0 = 0;
//     var_mu0_inv = 0;
//     for (int j = 0; j < Yproj.cols(); j++) {
//       mu0 += (double)Yproj.coeffRef(i, j) * sigma_sq_inv.coeffRef(j);
//       if (Yproj.coeffRef(i, j) != 0) {
// 	var_mu0_inv += (double)sigma_sq_inv.coeffRef(j);
//       }
//     }
//     _eta.coeffRef(i) = (scalar_type)mu0;
//     mu0 /= var_mu0_inv;
//     logpr1 = -0.5 * (mu0 - __mu.coeffRef(i)) *
//       (mu0 - __mu.coeffRef(i)) * var_mu0_inv;
//     logpr0 = -0.5 * mu0 * mu0 * var_mu0_inv;
//     prior_pr1 = extra_distributions::std_normal_cdf(
//       (double)_gamma.coeffRef(i) );
//     unnorm_post_pr1 = std::exp(logpr1) * prior_pr1;
//     unnorm_post_pr0 = std::exp(logpr0) * (1 - prior_pr1);
//     Pr_delta_1 = (unnorm_post_pr1 + eps0) /
//       (unnorm_post_pr1 + unnorm_post_pr0 + 2 * eps0);
//     if (isnan(Pr_delta_1))
//       Pr_delta_1 = 0;
//     _delta.coeffRef(i) = 0;
//     if (_Uniform(stickygpm::rng()) < (scalar_type)Pr_delta_1)
//       _delta.coeffRef(i) = 1;
//   }
//   //
// };






template< typename RealType >
void stickygpm::projected_gp_parameters
<RealType>::_update_mu(
  const typename
    stickygpm::projected_gp_parameters<RealType>::matrix_type& Yproj,
  const typename
    stickygpm::projected_gp_parameters
    <RealType>::vector_type& sigma_sq_inv
) {
  const scalar_type sum_of_sigma_sq_inv = sigma_sq_inv.sum();
  _sample_xi();
  // _Vmu_inv = sum_of_sigma_sq_inv *
  //   _XCor * _delta.asDiagonal() * _XCor.adjoint() + _Ks;
  _Vmu_inv = sum_of_sigma_sq_inv * _XCor * _XCor.adjoint() + _Ks;
  _llt_Vmu = _Vmu_inv.llt();
  // _eta set in _update_gamma_delta(...) above
  _eta = Yproj * sigma_sq_inv;  // <- will need to change w/ clustering
  // std::cout << "  mean(eta) = " << _eta.mean() << std::endl;
  // _mu_k = _llt_Vmu.solve( (_XCor * _delta.asDiagonal() * _eta) +
  // 			_llt_Vmu.matrixL() * _xi);
  _mu_k = _llt_Vmu.solve( (_XCor * _eta) + _llt_Vmu.matrixL() * _xi );
  // std::cout << "  mean(mu_*) = " << _mu_k.mean() << std::endl;
  // __mu = (_mu_k.transpose() * _XCor).transpose();
  // __mu = _delta.asDiagonal() * (_mu_k.transpose() * _XCor).transpose();
  // __mu = ( _mu_k.transpose() * _XCor ).transpose();
};


template< typename RealType >
void stickygpm::projected_gp_parameters
<RealType>::_update_mu(
  const typename
    stickygpm::projected_gp_parameters<RealType>::matrix_type& Yproj,
  const typename
    stickygpm::projected_gp_parameters
    <RealType>::vector_type& sigma_sq_inv,
  const std::vector<int>& subset
) {
  scalar_type sum_of_sigma_sq_inv = 0;
  _eta.setZero();
  _sample_xi();
  if ( subset.empty() ) {

    _llt_Vmu = _Ks.llt();
    _mu_k = _llt_Vmu.solve( _llt_Vmu.matrixL() * _xi );

    // Hacky fix:
    scalar_type sd_ = (scalar_type)stickygpm::vsd(
      ( _mu_k.transpose() * _XCor ).transpose().eval() );
    _mu_k /= sd_;
    // ^^ Sparse _XCor is a horrible approximation for the true
    //    cross-correlation matrix when there is no likelihood
    //    information to overcome. To "fix," force unit variance
    //    in the product _XCor.adjugate() * _mu_k. Seems to work
    //    ok for now in practice
    
  }
  else {

    for (std::vector<int>::const_iterator it = subset.cbegin();
	 it != subset.cend(); ++it) {
      sum_of_sigma_sq_inv += sigma_sq_inv[*it];
      _eta += Yproj.col(*it) * sigma_sq_inv[*it];
    }
    _Vmu_inv = sum_of_sigma_sq_inv * _XCor * _XCor.adjoint() + _Ks;
    _llt_Vmu = _Vmu_inv.llt();
    _mu_k = _llt_Vmu.solve( (_XCor * _eta) +
      _llt_Vmu.matrixL() * _xi );
    
  }
};








template< typename RealType >
void stickygpm::projected_gp_parameters<RealType>::update(
  const typename stickygpm::projected_gp_parameters
    <RealType>::matrix_type& Yproj,
  const typename stickygpm::projected_gp_parameters
    <RealType>::vector_type& sigma_sq_inv
) {
  // _update_gamma_delta(Yproj, sigma_sq_inv);
  _update_mu(Yproj, sigma_sq_inv);
};




template< typename RealType >
void stickygpm::projected_gp_parameters<RealType>::update(
  const matrix_type& Yproj,
  const vector_type& sigma_sq_inv,
  const std::vector<int>& subset
) {
  _update_mu(Yproj, sigma_sq_inv, subset);
};








template< typename RealType >
double stickygpm::projected_gp_parameters<RealType>::log_likelihood(
  const typename
    stickygpm::projected_gp_parameters<RealType>::matrix_type&
    Yproj,
  const typename
    stickygpm::projected_gp_parameters<RealType>::vector_type&
    sigma_sq_inv,
  const int which
) const {
  return -0.5 * (double)sigma_sq_inv.coeffRef(which) *
    stickygpm::vdot((Yproj.col(which) - mu()).eval());
};



template< typename RealType >
typename stickygpm::projected_gp_parameters
  <RealType>::vector_type
stickygpm::projected_gp_parameters<RealType>::mu() const {
  // return _delta.asDiagonal() * __mu;
  // return __mu;
  // vector_type mu_ = ( _mu_k.transpose() * _XCor ).transpose();
  // return mu_;
  return ( _mu_k.transpose() * _XCor ).transpose();
};



template< typename RealType >
const typename stickygpm::projected_gp_parameters
  <RealType>::sparse_matrix_type&
stickygpm::projected_gp_parameters<RealType>::xcor_matrix() const {
  return _XCor;
};


// template< typename RealType >
// const typename stickygpm::projected_gp_parameters
//   <RealType>::vector_type&
// stickygpm::projected_gp_parameters<RealType>::delta() const {
//   return _delta;
// };




#endif  // _STICKYGPM_PROJECTED_GP_PARAMETERS_



