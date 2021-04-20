
#include <algorithm>
#include <cassert>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <iostream>
#include <nifti1_io.h>
#include <random>
#include <stdexcept>
#include <vector>

#include "stickygpm/covariance_functors.h"
#include "stickygpm/eigen_slicing.h"
#include "stickygpm/kernels.h"
#include "stickygpm/utilities.h"
#include "stickygpm/nifti_manipulation.h"
#include "stickygpm/voxel_neighborhoods.h"


#ifndef _STICKYGPM_KNOTS_
#define _STICKYGPM_KNOTS_


namespace stickygpm {


  

  

  template< typename RealType = float >
  Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>
  get_knot_positions_uniform(const ::nifti_image* const mask,
			     const int ngrid = 2048,
			     const bool jitter = true
  ) {
    if (ngrid <= 0)
      throw std::domain_error("Dimension of knot grid must be >= 0");
    std::uniform_real_distribution<RealType> Uniform(-0.1, 0.1);
    stickygpm::qform_type Q = stickygpm::qform_matrix(mask);
    Eigen::MatrixXi ijk = stickygpm::get_nonzero_indices(mask);
    std::vector<int> row_index(ijk.rows());
    for ( int i = 0; i < (int)row_index.size(); i++ ) {
      row_index[i] = i;
    }
    std::shuffle(row_index.begin(), row_index.end(), stickygpm::rng());
    Eigen::VectorXi rows_ = Eigen::Map<Eigen::VectorXi>(
      row_index.data(), std::min( ngrid, (int)row_index.size() ));
    ijk.conservativeResize(ijk.rows(), ijk.cols() + 1);
    ijk.col(ijk.cols() - 1) = Eigen::VectorXi::Ones(ijk.rows());
    Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> Knots =
      stickygpm::nullary_index(
        ( ijk.cast<float>() * Q.transpose() ).eval(),
	rows_, Eigen::VectorXi::LinSpaced(3, 0, 2))
      .template cast<RealType>();
    if ( jitter ) {
      for (int i = 0; i < Knots.rows(); i++) {
	for (int j = 0; j < Knots.cols(); j++)
	  Knots.coeffRef(i, j) += Uniform(stickygpm::rng());
      }
    }
    // Eigen::VectorXi::LinSpaced(ngrid, 0, ijk.rows() - 1),
    return Knots;
  };








  template< typename RealType >
  RealType minimax_knot_distance(
    const ::nifti_image* const nii,
    const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>& Knots
  ) {
    typedef typename
      Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>
      matrix_type;
    matrix_type Coord = stickygpm::get_nonzero_indices(nii)
      .cast<RealType>();
    Coord.conservativeResize(Coord.rows(), Coord.cols() + 1);
    Coord.col(Coord.cols() - 1) = Eigen::VectorXf::Ones(Coord.rows());
    Coord = ( Coord * stickygpm::qform_matrix(nii).transpose() ).eval();
    Coord.conservativeResize(Coord.rows(), Coord.cols() - 1);
    RealType _d, _min_d_knots, minimax_distance = 0;
    for (int i = 0; i < Coord.rows(); i++) {
      _min_d_knots = 1e6;
      for (int j = 0; j < Knots.rows(); j++) {
	_d = (Coord.row(i) - Knots.row(j)).norm();
	_min_d_knots = std::min(_min_d_knots, _d);
      }
      minimax_distance = std::max(minimax_distance, _min_d_knots);
    }
    return minimax_distance;
  };






  
  

  template< typename RealType >
  Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>
  knot_covariance_matrix(
    const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>&
      Knots,
    const stickygpm::covariance_functor<RealType>& cov
  ) {
    typedef typename
      Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>
      matrix_type;
    const RealType c0 = cov(0);
    matrix_type Sigma( Knots.rows(), Knots.rows() );
    RealType c, distance;
    Sigma.coeffRef(0, 0) = c0;
    for (int i = 1; i < Knots.rows(); i++) {
      for (int j = 0; j < i; j++) {
	distance = ( Knots.row(i) - Knots.row(j) ).norm();
	c = cov( distance );
	Sigma(i, j) = c;
	Sigma(j, i) = c;
      }
      Sigma.coeffRef(i, i) = c0;
    }
    return Sigma;
  };





  
  template< typename RealType >
  Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>
  knot_covariance_matrix(
    const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>&
      Knots,
    const stickygpm::covariance_functor<RealType>& cov,
    // const Eigen::Matrix<RealType, Eigen::Dynamic, 1>& diag
    const RealType nugget
  ) {
    // assert( diag.size() == Knots.rows() && "Misaligned arguments" );
    assert( nugget >= 0 && "Supplied a negative nugget variance" );
    typedef typename
      Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>
      matrix_type;
    const RealType c0 = cov(0) + nugget;
    matrix_type Sigma( Knots.rows(), Knots.rows() );
    RealType c, distance;
    // Sigma.coeffRef(0, 0) = c0 + diag.coeffRef( 0 );
    Sigma.coeffRef(0, 0) = c0;
    for (int i = 1; i < Knots.rows(); i++) {
      for (int j = 0; j < i; j++) {
	distance = ( Knots.row(i) - Knots.row(j) ).norm();
	c = cov( distance );
	Sigma(i, j) = c;
	Sigma(j, i) = c;
      }
      // Sigma.coeffRef(i, i) = c0 + diag.coeffRef( i );
      Sigma.coeffRef(i, i) = c0;
    }
    return Sigma;
  };




  // template< typename RealType >
  // Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>
  // knot_covariance_matrix(
  //   const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>&
  //     Knots,
  //   const stickygpm::covariance_functor<RealType>& cov,
  //   // const Eigen::Matrix<RealType, Eigen::Dynamic, 1>& diag
  //   const Eigen::Matrix<RealType, Eigen::Dynamic, 1>&
  //     sill
  // ) {
  //   typedef typename
  //     Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>
  //     matrix_type;
  //   matrix_type Sigma( Knots.rows(), Knots.rows() );
  //   RealType c, distance;
  //   Sigma.coeffRef(0, 0) = sill.coeff( 0 );
  //   for (int i = 1; i < Knots.rows(); i++) {
  //     for (int j = 0; j < i; j++) {
  // 	distance = ( Knots.row(i) - Knots.row(j) ).norm();
  // 	c = cov( distance );
  // 	Sigma(i, j) = c;
  // 	Sigma(j, i) = c;
  //     }
  //     Sigma.coeffRef(i, i) = sill.coeff(i);
  //   }
  //   return Sigma;
  // };




  



  
  template< typename RealType >
  Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>
  covariance_matrix_gradient(
    const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>&
      locations,
    const stickygpm::covariance_functor<RealType>& cov,
    const int param_j
  ) {
    assert( param_j >= 0 && param_j < cov.param_size() &&
	    "Parameter index out of scope" );
    typedef typename
      Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>
      matrix_type;
    const RealType grad0 = cov.gradient(0)[ param_j ];
    const int nloc = locations.rows();
    matrix_type Sigma_prime( nloc, nloc );
    RealType distance;
    std::vector<RealType> grad;
    Sigma_prime.coeffRef(0, 0) = grad0;
    for (int i = 1; i < nloc; i++) {
      for (int j = 0; j < i; j++) {
	distance = ( locations.row(i) - locations.row(j) ).norm();
	grad = cov.gradient( distance );
	Sigma_prime(i, j) = grad[ param_j ];
	Sigma_prime(j, i) = grad[ param_j ];
      }
      Sigma_prime.coeffRef(i, i) = grad0;
    }
    return Sigma_prime;
  };






  template< typename RealType >
  Eigen::SparseMatrix<RealType, Eigen::RowMajor>
  knn_basis_matrix2(
    const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>&
    Locations,
    const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>&
    Knots,
    const stickygpm::covariance_functor<RealType>& cov,
    const int k = 2000
  ) {
    // Version to limit the number of voxels per knot
    typedef RealType scalar_type;
    typedef typename Eigen::SparseMatrix<scalar_type, Eigen::RowMajor>
      sparse_matrix_type;
    typedef typename Eigen::Triplet<scalar_type> triplet_type;
    typedef typename Eigen::Matrix<scalar_type, Eigen::Dynamic, 1>
      vector_type;
    assert( Locations.cols() == Knots.cols() &&
	    "Incompatible dimensions: Location/Knot coordinates" );
    const int nknots = Knots.rows();
    const int nvox = Locations.rows();
    const scalar_type eps0 = 1e-4;
    scalar_type c;
    vector_type squared_distances( nvox );
    sparse_matrix_type PMat( nvox, nknots );
    std::vector<int> ord( nvox );
    std::vector<triplet_type> triplet_list;
    triplet_list.reserve( nknots * k );
    for ( int j = 0; j < Knots.rows(); j++ ) {
      // Find k closest voxels
      squared_distances = (Locations.rowwise() - Knots.row(j))
	.rowwise().squaredNorm();
      std::iota( ord.begin(), ord.end(), 0 );
      std::sort( ord.begin(), ord.end(),
		 [&](int a, int b) -> bool {
		   return squared_distances.coeffRef(a) <
		     squared_distances.coeffRef(b);
		 });
      // Compute covariance between knot j and k closest voxels
      for ( int i = 0; i < k; i++ ) {
	c = cov( std::sqrt(squared_distances.coeffRef(ord[i]) + eps0) );
	// c = cov(std::sqrt( squared_distances.coeffRef(ord[i]) ));
	triplet_list.push_back(
          triplet_type( ord[i], j, c )
        );
      }
    }
    PMat.setFromTriplets( triplet_list.begin(), triplet_list.end() );
    return PMat;
  };





  template< typename RealType >
  Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>
  dense_basis_matrix(
    const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>&
    Locations,
    const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>&
    Knots,
    const stickygpm::covariance_functor<RealType>& cov
  ) {
    typedef RealType scalar_type;
    typedef typename Eigen::Matrix<
      scalar_type, Eigen::Dynamic, Eigen::Dynamic >
      matrix_type;
    assert( Locations.cols() == Knots.cols() &&
	    "Incompatible dimensions: Location/Knot coordinates" );
    matrix_type BMat( Locations.rows(), Knots.rows() );
    scalar_type distance;
    for ( int i = 0; i < Locations.rows(); i++ ) {
      // Compute covariance between location i knots
      for ( int j = 0; j < Knots.rows(); j++ ) {
	distance = (Knots.row(j) - Locations.row(i)).norm();
	BMat.coeffRef(i, j) = cov( distance );
      }
    }
    return BMat;
  };
  







  template< typename RealType >
  Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>
  projection_inner_product_matrix(
    const ::nifti_image* const mask,
    const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>& Knots,
    const std::vector<RealType>& rbf_params
  ) {
    typedef typename
      Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>
      matrix_type;
    typedef typename
      Eigen::Matrix<RealType, Eigen::Dynamic, 1>
      vector_type;
    const matrix_type xyz = stickygpm::get_nonzero_xyz(mask)
      .template cast<RealType>();
    const int nknots = Knots.rows();
    const int nloc = xyz.rows();
    matrix_type PtP = matrix_type::Zero(nknots, nknots);
    vector_type dists;
    // A' * A = \sum_i a_i * a_i'
    // Compute lower triangle (+ diagonal)
    for (int i = 0; i < nloc; i++) {
      dists = (Knots.rowwise() - xyz.row(i)).rowwise().norm();
      for (int j = 0; j < dists.size(); j++) {
	dists.coeffRef(j) = stickygpm::kernels::rbf(dists.coeffRef(j),
          rbf_params[1], rbf_params[2], rbf_params[0]);
	for (int k = 0; k <= j; k++) {
	  PtP.coeffRef(j, k) += dists.coeffRef(j) * dists.coeffRef(k);
	}
      }
    }
    // Assign upper triangle (no diagonal)
    for (int i = 1; i < nknots; i++) {
      for (int j = 0; j < i; j++) {
    	PtP.coeffRef(j, i) = PtP.coeffRef(i, j);
      }
    }
    return PtP;
  };




  template< typename RealType >
  void make_knot_image(
    ::nifti_image* const mask,  /*!< Modified */
    const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>& Knots
  ) {
    typedef typename
      Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>
      matrix_type;
    const stickygpm::qform_type Qinv =
      stickygpm::qform_matrix(mask).inverse();
    matrix_type xyz = Knots;
    xyz.conservativeResize(xyz.rows(), xyz.cols() + 1);
    xyz.col(xyz.cols() - 1) = matrix_type::Ones(xyz.rows(), 1);
    Eigen::MatrixXi ijk = ( xyz * Qinv.transpose() )
      .template cast<int>();
    int stride;
    float* const mask_ptr = (float*)mask->data;
    stickygpm::set_zero( mask );
    for (int i = 0; i < ijk.rows(); i++) {
      stride = mask->nx * mask->ny * ijk(i, 2) +
	mask->nx * ijk(i, 1) + ijk(i, 0);
      *(mask_ptr + stride) = (float)1.0;
    }
  };
  
  
  
}
// end - namespace stickygpm



#endif  // _STICKYGPM_KNOTS_


