
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
    for (int i = 0; i < row_index.size(); i++) {
      row_index[i] = i;
    }
    std::shuffle(row_index.begin(), row_index.end(), stickygpm::rng());
    Eigen::VectorXi rows_ = Eigen::Map<Eigen::VectorXi>(
      row_index.data(), std::min(ngrid, (int)row_index.size()));
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

  
  
  
}
// end - namespace stickygpm



#endif  // _STICKYGPM_KNOTS_











  
  

  // template< typename RealType >
  // Eigen::SparseMatrix<RealType, Eigen::RowMajor>
  // knn_basis_matrix(
  //   const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>&
  //   Locations,
  //   const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>&
  //   Knots,
  //   const stickygpm::covariance_functor<RealType>& cov,
  //   const int k = 50
  // ) {
  //   // Version to limit the number of voxels per knot
  //   typedef RealType scalar_type;
  //   typedef typename Eigen::Matrix<
  //     scalar_type, Eigen::Dynamic, Eigen::Dynamic >
  //     matrix_type;
  //   typedef typename Eigen::SparseMatrix<scalar_type, Eigen::RowMajor>
  //     sparse_matrix_type;
  //   typedef typename Eigen::Triplet<scalar_type> triplet_type;
  //   typedef typename Eigen::Matrix<scalar_type, Eigen::Dynamic, 1>
  //     vector_type;
  //   assert( Locations.cols() == Knots.cols() &&
  // 	    "Incompatible dimensions: Location/Knot coordinates" );
  //   const int nknots = Knots.rows();
  //   const scalar_type eps0 = 1e-4;
  //   const matrix_type Sigma_star =
  //     stickygpm::knot_covariance_matrix( Knots, cov );
  //   Eigen::VectorXi knots_index_( k );
  //   vector_type squared_distances( nknots );
  //   vector_type covar_with_knots( k );
  //   vector_type pmat_row( k );
  //   sparse_matrix_type PMat(Locations.rows(), nknots);
  //   std::vector<int> ord( nknots );
  //   std::vector<triplet_type> triplet_list;
  //   triplet_list.reserve( Locations.rows() * nknots );
  //   for ( int i = 0; i < Locations.rows(); i++ ) {
  //     // Find k closest knots
  //     squared_distances = (Knots.rowwise() - Locations.row(i))
  // 	.rowwise().squaredNorm();
  //     std::iota( ord.begin(), ord.end(), 0 );
  //     std::sort( ord.begin(), ord.end(),
  // 		 [&](int a, int b) -> bool {
  // 		   return squared_distances.coeffRef(a) <
  // 		     squared_distances.coeffRef(b);
  // 		 });
  //     // Compute covariance between location i and k closest knots
  //     for ( int j = 0; j < k; j++ ) {
  // 	covar_with_knots.coeffRef(j) = 
  // 	  cov( std::sqrt(squared_distances.coeffRef(j) + eps0) );
  // 	knots_index_.coeffRef(j) = ord[j];
  //     }
  //     // Solve system and add sparse row to data
  //     pmat_row = stickygpm::nullary_index(
  //       Sigma_star, knots_index_, knots_index_
  //     ).colPivHouseholderQr().solve( covar_with_knots );
  //     for ( int j = 0; j < k; j++ ) {
  // 	triplet_list.push_back(
  //         triplet_type( i, ord[j], pmat_row.coeffRef(j) )
  //       );
  //     }
  //   }
  //   PMat.setFromTriplets( triplet_list.begin(), triplet_list.end() );
  //   return PMat;
  // };





  

  // template< typename RealType = float >
  // struct sparse_matrix_data {
  //   std::vector<RealType> _Data;
  //   std::vector<int> column_indices;
  //   std::vector<int> cum_row_counts;
  //   int ncol;
  //   int nrow;
  // };

  



  // template< typename RealType = float, typename ImageType = float >
  // stickygpm::sparse_matrix_data<RealType>
  // get_sparse_crosscorrelation_data(
  //   const ::nifti_image* const mask,
  //   const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>& Knots,
  //   const std::vector<RealType>& rbf_params,
  //   const RealType k = 1.4
  // ) {
  //   const int nvox_mask_ = (int)mask->nvox;
  //   const ImageType* const mask_data_ptr = (ImageType*)mask->data;
  //   const float knot_mmx_dist_ =
  //     stickygpm::minimax_knot_distance(mask, Knots);
  //   const float kernel_fwhm_ =
  //     (float)stickygpm::kernels::rbf_bandwidth_to_fwhm(
  //       rbf_params[1], rbf_params[2]);


  //   const std::vector<int> mask_nonzero_flat_index_map =
  //     stickygpm::get_nonzero_flat_index_map(mask);

  //   const stickygpm::qform_type Q = stickygpm::qform_matrix(mask);
  //   const stickygpm::qform_type Qinv = Q.inverse();
    
  //   stickygpm::neighborhood_ball<RealType> Ball =
  //     stickygpm::neighborhood_perturbation<RealType>(
  //       stickygpm::qform_matrix(mask),
  // 	k * std::max(knot_mmx_dist_, kernel_fwhm_)
  //     );

  //   stickygpm::sparse_matrix_data<RealType> smd;
  //   smd.cum_row_counts.resize(Knots.rows() + 1);
  //   smd.cum_row_counts[0] = 0;
  //   smd.ncol = stickygpm::count_nonzero_voxels(mask);
  //   smd.nrow = Knots.rows();

  //   std::vector<int> voxel_indices;
  //   std::vector<RealType> cross_covariance;
  //   Eigen::Vector4f knot_position = Eigen::Vector4f::Ones();
  //   Eigen::Vector3f voxel_xyz;
  //   Eigen::Vector3i current_position, knot_index;
  //   int voxel_offset;
  //   int row_count = 0, empty_row_count = 0;
  //   RealType voxel_value;

  //   for (int i = 0; i < Knots.rows(); i++) {
  //     row_count = 0;
      
  //     voxel_indices.clear();
  //     cross_covariance.clear();
  //     voxel_indices.reserve(Ball.perturbations.rows());
  //     cross_covariance.reserve(Ball.perturbations.rows());

  //     knot_position.head<3>() = Knots.row(i).transpose()
  // 	.template cast<float>();
  //     knot_index = (Qinv * knot_position).head<3>()
  // 	.template cast<int>();

      
  //     for (int j = 0; j < Ball.perturbations.rows(); j++) {
  // 	current_position = knot_index +
  // 	  Ball.perturbations.row(j).transpose();
  // 	voxel_xyz =
  // 	  Q.block<3,3>(0, 0) * current_position.template cast<float>() +
  // 	  Q.block<3,1>(0, 3);
  // 	voxel_offset = current_position[2] * mask->ny * mask->nx +
  // 	  current_position[1] * mask->nx + current_position[0];

  // 	if (voxel_offset >= 0 && voxel_offset < nvox_mask_) {
  // 	  voxel_value = *(mask_data_ptr + voxel_offset);
  // 	  if (!(isnan(voxel_value) || voxel_value == 0)) {
  // 	    if (mask_nonzero_flat_index_map[voxel_offset] == -1) {
  // 	      std::cerr << "Warning!\n";
  // 	    }
  // 	    voxel_indices.push_back(mask_nonzero_flat_index_map[voxel_offset]);
  // 	    // voxel_indices.push_back(voxel_offset);
  // 	    cross_covariance.push_back(
  // 				       // stickygpm::kernels::rbf(Ball.distances[j],
  //               stickygpm::kernels::rbf((voxel_xyz - Knots.row(i).transpose()).norm(),
  //               rbf_params[1], rbf_params[2], rbf_params[0])
  //             );
  // 	    row_count++;
  // 	  }
  // 	}  // if (voxel_offset >= 0 && voxel_offset < nvox_mask_) {
  //     }  // for (int j = 0; j < Ball.perturbations.rows(); j++) {

  //     voxel_indices.shrink_to_fit();
  //     cross_covariance.shrink_to_fit();
  //     smd.cum_row_counts[i + 1] = smd.cum_row_counts[i];
      
  //     if (row_count == 0) {
  // 	empty_row_count++;
  //     }
  //     else {
  // 	smd._Data.insert(smd._Data.end(), cross_covariance.begin(),
  //         cross_covariance.end());
  // 	smd.column_indices.insert(smd.column_indices.end(),
  //         voxel_indices.begin(), voxel_indices.end());
  // 	smd.cum_row_counts[i + 1] += cross_covariance.size();
  //     }
  //   }  // for (int i = 0; i < Knots.rows(); i++) {

  //   if (empty_row_count > 0) {
  //     std::cerr << "Warning: " << empty_row_count
  // 		<< " knot/data cross correlation rows were all zero"
  // 		<< std::endl;
  //   }

  //   return smd;
  // };
  





