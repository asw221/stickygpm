
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <iostream>
#include <memory>
#include <nifti1_io.h>
#include <random>
#include <vector>

#include "stickygpm/covariance_functors.h"
#include "stickygpm/eigen_slicing.h"
#include "stickygpm/stickygpm_regression_data.h"
#include "stickygpm/nifti_manipulation.h"
#include "stickygpm/utilities.h"
#include "stickygpm/voxel_neighborhoods.h"


#ifndef _STICKYGPM_NNGP_REGRESSION_
#define _STICKYGPM_NNGP_REGRESSION_


namespace stickygpm {


  template< typename T, typename CovT >
  class nngp_regression {
  public:
    typedef T scalar_type;
    typedef CovT covariance_type;
    typedef typename
    Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>
    matrix_type;
    typedef typename
    Eigen::SparseMatrix<scalar_type>
    sparse_matrix_type;
    typedef typename
    Eigen::Matrix<scalar_type, Eigen::Dynamic, 1>
    vector_type;

    class shared_data {
    public:
      typedef typename Eigen::Triplet<T> triplet_type;
      
      shared_data(
        const ::nifti_image* const mask,
	const CovT& cov,
	const T neighborhood
      );

      matrix_type covariance_matrix(const matrix_type& locations) const;

      const sparse_matrix_type& matrixIA() const;
      const vector_type& vectorDinv() const;

      const Eigen::SimplicialLDLT<sparse_matrix_type>& gram_ldlt() const;
      Eigen::SimplicialLDLT<sparse_matrix_type>& gram_ldlt();
      
    private:
      CovT _Covf;
      sparse_matrix_type _I_minus_A;
      vector_type _di;

      Eigen::SimplicialLDLT<sparse_matrix_type> _ldlt_;
    };
    // class shared_data


    nngp_regression();
    nngp_regression(const stickygpm::stickygpm_regression_data<T>& data);
    nngp_regression(const nngp_regression<T, CovT>& other);

    // void link_to_shared_data(const shared_data& data);
    void link_to_shared_data(const std::shared_ptr<shared_data> data);
    void update(
      const stickygpm::stickygpm_regression_data<T>& data,
      const vector_type& sigma_sq_inv
    );
    void sample_from_prior();

    const matrix_type& beta() const;
    vector_type beta(const int col) const;
    
    double log_likelihood(
      const stickygpm::stickygpm_regression_data<T>& data,
      const vector_type& sigma_sq_inv,
      const int i
    ) const;

    operator bool() const;
    bool operator !() const;
    

  private:
    std::shared_ptr<shared_data> _p_data;
    matrix_type _beta;    // Regression coefficients with NNGP prior
    vector_type _rgauss;  // Std Gaussian samples - (dim _beta);

    void _draw_gaussian();
  };
  // class nngp_regression




};
// namespace stickygpm






template< typename T, typename CovT >
stickygpm::nngp_regression<T, CovT>::nngp_regression() {
  _beta = matrix_type::Zero(1, 1);
  _rgauss = vector_type::Zero(1);
};


template< typename T, typename CovT >
stickygpm::nngp_regression<T, CovT>::nngp_regression(
  const stickygpm::stickygpm_regression_data<T>& data
) {
  _beta = matrix_type::Zero( data.Y().rows(), data.X().cols() );
  _rgauss = vector_type::Zero( _beta.rows() );
};


template< typename T, typename CovT >
stickygpm::nngp_regression<T, CovT>::nngp_regression(
  const nngp_regression<T, CovT>& other
) {
  _p_data = other._p_data;
  _beta = other._beta;
  _rgauss = vector_type::Zero( _beta.rows() );
};


template< typename T, typename CovT >
void stickygpm::nngp_regression<T, CovT>::link_to_shared_data(
  // const stickygpm::nngp_regression<T>::shared_data& data
  const std::shared_ptr<stickygpm::nngp_regression<T, CovT>::shared_data> p_data
) {
  // _p_data = std::make_shared<const shared_data>( data );
  _p_data = p_data;
};


template< typename T, typename CovT >
void stickygpm::nngp_regression<T, CovT>::update(
  const stickygpm::stickygpm_regression_data<T>& data,
  const typename stickygpm::nngp_regression<T, CovT>::vector_type&
    sigma_sq_inv
) {
  assert( _p_data && "Pointer to prior LDLT data is null" );
  //
};


template< typename T, typename CovT >
void stickygpm::nngp_regression<T, CovT>::sample_from_prior() {
  assert( _p_data && "Pointer to prior LDLT data is null" );

  double run_t = 0;
  auto start_t = std::chrono::high_resolution_clock::now();

  sparse_matrix_type Vi = 
    _p_data->matrixIA().adjoint() *
    _p_data->vectorDinv().asDiagonal() *
    _p_data->matrixIA();
  // Vi += vector_type::Constant( _beta.rows(), (scalar_type)0.001).asDiagonal();
  std::cout << "\tMatrix V^-1 has " << Vi.nonZeros() << " non-zero elements"
	    << std::endl;
  std::cout << "Upper Left:\n"
	    << _p_data->matrixIA().topLeftCorner(10, 10)
	    << "\nBottom Right:\n"
	    << _p_data->matrixIA().bottomRightCorner(10, 10)
	    << std::endl;

  std::cout << _p_data->vectorDinv().head(20).transpose() << "\n"
	    << _p_data->vectorDinv().tail(20).transpose() << "\n"
	    << std::endl;
  
  _p_data->gram_ldlt().factorize( Vi );
  auto stop_t = std::chrono::high_resolution_clock::now();
  auto diff_t = std::chrono::duration_cast<std::chrono::microseconds>(stop_t - start_t);
  std::cout << "\tDecomposition took "
	    << ((double)diff_t.count() / 1e6) << " (sec)"
	    << std::endl;
  
  for ( int j = 0; j < _beta.cols(); j++ ) {
    _draw_gaussian();

    // std::cout << (_p_data->gram_ldlt().matrixL() * _rgauss).head(10)
    // 	      << "\n\n";
    
    start_t = std::chrono::high_resolution_clock::now();
    // _beta.col(j) = _p_data->gram_ldlt().solve(
    //   _p_data->gram_ldlt().matrixL() *
    //   ( _p_data->gram_ldlt().vectorD().cwiseSqrt().asDiagonal() *
    // 	_rgauss )
    // );
    _beta.col(j) = _p_data->gram_ldlt().solve(
      _p_data->matrixIA().adjoint() *
      ( _p_data->vectorDinv().cwiseSqrt().asDiagonal() *
    	_rgauss )
    );
    stop_t = std::chrono::high_resolution_clock::now();
    diff_t = std::chrono::duration_cast<std::chrono::microseconds>(stop_t - start_t);
    run_t += (double)diff_t.count() / 1e6;
  }

  std::cout << _beta.colwise().mean()
  	    << std::endl
  	    << (_beta.colwise().squaredNorm() / _beta.rows() -
  		_beta.colwise().mean().array().pow(2).matrix())
  	    << std::endl;
  
  std::cout << "\tSparse solver took on average "
	    << (run_t / _beta.cols()) << " (sec)"
	    << std::endl;
};
// sample_from_prior()


template< typename T, typename CovT >
const typename stickygpm::nngp_regression<T, CovT>::matrix_type&
stickygpm::nngp_regression<T, CovT>::beta() const {
  return _beta;
};


template< typename T, typename CovT >
typename stickygpm::nngp_regression<T, CovT>::vector_type
stickygpm::nngp_regression<T, CovT>::beta(
  const int col
) const {
  return _beta.col( col );
};


template< typename T, typename CovT >    
double stickygpm::nngp_regression<T, CovT>::log_likelihood(
  const stickygpm::stickygpm_regression_data<T>& data,
  const typename stickygpm::nngp_regression<T, CovT>::vector_type&
    sigma_sq_inv,
  const int i
) const {
  assert( data.Y().cols() == sigma_sq_inv.size() &&
	  "nngp_regression:log_likelihood : data/parmeter mismatch" );
  //
};




template< typename T, typename CovT >    
stickygpm::nngp_regression<T, CovT>::operator bool() const {
  return static_cast<bool>(_p_data);
};


template< typename T, typename CovT >    
bool stickygpm::nngp_regression<T, CovT>::operator !() const {
  return !static_cast<bool>(_p_data);
};



template< typename T, typename CovT >
void stickygpm::nngp_regression<T, CovT>::_draw_gaussian() {
  std::normal_distribution<scalar_type> Gaussian(0, 1);
  for ( int i = 0; i < _rgauss.size(); i++ ) {
    _rgauss.coeffRef(i) = Gaussian( stickygpm::rng() );
  }
};




// --- nngp_regression<T>::shared_data -------------------------------


template< typename T, typename CovT >
stickygpm::nngp_regression<T, CovT>::shared_data::shared_data(
  const ::nifti_image* const mask,
  const CovT& cov,
  const T neighborhood
) {
  _Covf = cov;
  const matrix_type xyz = stickygpm::get_nonzero_xyz(mask)
    .template cast<scalar_type>();
  const Eigen::VectorXi all_cols_ =
    Eigen::VectorXi::LinSpaced(xyz.cols(), 0, xyz.cols() - 1);
  scalar_type dx, dy, dz, distance;
  std::vector<triplet_type> triplet_list;
  // std::vector<int> neighbor_indices;
  // std::vector<T> csub_raw;
  int prev_nsize = 1, reserve_n;
  
  _di = vector_type::Constant( xyz.rows(), 1 / cov(0) );
  std::cout << "Cov(0) = " << _Covf(0) << std::endl;

  std::cout << "\tReserving triplets\n" << std::flush;
  reserve_n =
    stickygpm::neighborhood_cardinality(mask, neighborhood) *
    3 / 5;
  reserve_n = (reserve_n < 1) ? 1 : reserve_n;
  triplet_list.reserve( xyz.rows() * reserve_n );
  triplet_list.push_back(
    triplet_type( 0, 0, static_cast<scalar_type>(1) )
  );

  std::cout << "\tLooping over lower triangle of I - A\n" << std::flush;
  // Loop over lower triangle of (I - A)
  for ( int i = 1; i < xyz.rows(); i++ ) {
    reserve_n = static_cast<int>( 1.1 * prev_nsize );
    std::vector<T> csub_raw;
    std::vector<int> neighbor_indices;
    csub_raw.reserve( reserve_n );
    neighbor_indices.reserve( reserve_n );
    for ( int j = 0; j < i; j++ ) {
      dx = std::abs( xyz.coeffRef(i, 0) - xyz.coeffRef(j, 0) );
      dy = std::abs( xyz.coeffRef(i, 1) - xyz.coeffRef(j, 1) );
      dz = std::abs( xyz.coeffRef(i, 2) - xyz.coeffRef(j, 2) );
      if ( dx <= neighborhood &&
	   dy <= neighborhood &&
	   dz <= neighborhood ) {
	distance = std::sqrt( dx * dx + dy * dy + dz * dz );
	if ( distance <= neighborhood ) {
	  csub_raw.push_back( cov(distance) );
	  neighbor_indices.push_back( j );
	}
      }
    }
    // end - for ( int j = 0; j < i; j++ )
    csub_raw.shrink_to_fit();
    neighbor_indices.shrink_to_fit();

    if ( !neighbor_indices.empty() ) {
      Eigen::VectorXi neighbors_ = Eigen::Map<Eigen::VectorXi>(
        neighbor_indices.data(),
	neighbor_indices.size()
      );
      vector_type c_sub = Eigen::Map<vector_type>(
        csub_raw.data(),
	csub_raw.size()
      );
      matrix_type Cov_sub = covariance_matrix(
        stickygpm::nullary_index(xyz, neighbors_, all_cols_)
      );
      vector_type c_tilde = Cov_sub.colPivHouseholderQr().solve( c_sub );
      // if ( i == 1500 ) {
      //   std::cout << "i = " << xyz.row(i) << "\n"
      // 		<< "N[i] =\n"
      // 		<< stickygpm::nullary_index(xyz, neighbors_, all_cols_)
      // 		<< "\n"
      // 		<< "Cs =\n" << Cov_sub << "\n"
      // 		<< "cs = " << c_sub.transpose() << "\n"
      // 		<< "c_tilde = " << c_tilde.transpose() << "\n"
      // 		<< "cs' c_tilde = " << (c_sub.transpose() * c_tilde)
      // 		<< std::endl;
      // }
      for ( int j = 0; j < c_tilde.size(); j++ ) {
        triplet_list.push_back(
          triplet_type( i, neighbor_indices[j], -c_tilde.coeffRef(j) )
        );
      }
      _di.coeffRef(i) = 1 / ( cov(0) - (c_sub.transpose() * c_tilde) );

      prev_nsize = static_cast<int>( neighbor_indices.size() );
      prev_nsize = (prev_nsize < 1) ? 1 : prev_nsize;
      // neighbor_indices.clear();
      // csub_raw.clear();
    }
    triplet_list.push_back(
      triplet_type( i, i, static_cast<scalar_type>(1) )
    );
  }
  // end - for ( int i = 1; i < xyz.rows(); i++ )

  std::cout << "\tConstructing sparse I - A\n" << std::flush;

  triplet_list.shrink_to_fit();
    
  _I_minus_A = sparse_matrix_type( xyz.rows(), xyz.rows() );
  _I_minus_A.setFromTriplets(
    triplet_list.begin(),
    triplet_list.end()
  );

  std::cout << "\tComputing decomposition\n" << std::flush;

  // Precompute LDLT decomposition pattern
  _ldlt_.analyzePattern(
    _I_minus_A.adjoint() * _di.asDiagonal() * _I_minus_A
  );
};





template< typename T, typename CovT >
typename stickygpm::nngp_regression<T, CovT>::matrix_type
stickygpm::nngp_regression<T, CovT>::shared_data::covariance_matrix(
  const typename stickygpm::nngp_regression<T, CovT>::matrix_type&
  locations
) const {
  const int nloc = locations.rows();
  const scalar_type c0 = _Covf(0);
  matrix_type Cov( nloc, nloc );
  scalar_type distance, c;
  Cov.coeffRef(0, 0) = c0;
  for ( int i = 1; i < nloc; i++ ) {
    for ( int j = 0; j < i; j++ ) {
      distance = (locations.row(i) - locations.row(j)).norm();
      c = _Covf(distance);
      Cov.coeffRef(i, j) = c;
      Cov.coeffRef(j, i) = c;
    }
    Cov.coeffRef(i, i) = c0;
  }
  return Cov;
};




template< typename T, typename CovT >
const typename stickygpm::nngp_regression<T, CovT>::sparse_matrix_type&
stickygpm::nngp_regression<T, CovT>::shared_data::matrixIA() const {
  return _I_minus_A;
};


template< typename T, typename CovT >
const typename stickygpm::nngp_regression<T, CovT>::vector_type&
stickygpm::nngp_regression<T, CovT>::shared_data::vectorDinv() const {
  return _di;
};



template< typename T, typename CovT >
const Eigen::SimplicialLDLT
<typename stickygpm::nngp_regression<T, CovT>::sparse_matrix_type>&
stickygpm::nngp_regression<T, CovT>::shared_data::gram_ldlt() const {
  return _ldlt_;
};


template< typename T, typename CovT >
Eigen::SimplicialLDLT
<typename stickygpm::nngp_regression<T, CovT>::sparse_matrix_type>&
stickygpm::nngp_regression<T, CovT>::shared_data::gram_ldlt() {
  return _ldlt_;
};



#endif  // _STICKYGPM_NNGP_REGRESSION_
