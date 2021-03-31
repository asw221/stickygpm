
#include <cmath>
#include <Eigen/Core>
#include <iostream>
#include <nifti1_io.h>
#include <vector>

#include "stickygpm/nifti_manipulation.h"



#ifndef _STICKYGPM_VOXEL_NEIGHBORHOODS_
#define _STICKYGPM_VOXEL_NEIGHBORHOODS_


namespace stickygpm {

  


  template< typename RealType = float >
  struct neighborhood_ball {
    typedef RealType scalar_type;
    typedef typename
      Eigen::Matrix<RealType, Eigen::Dynamic, 1> vector_type;
    Eigen::MatrixXi perturbations;
    vector_type distances;
  };




  template< typename RealType = int >
  Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>
  expand_grid(
    const Eigen::Matrix<RealType, Eigen::Dynamic, 1>& A,
    const Eigen::Matrix<RealType, Eigen::Dynamic, 1>& B
  ) {
    typedef typename
      Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>
      matrix_type;
    typedef typename Eigen::Matrix<RealType, Eigen::Dynamic, 1>
      vector_type;
    const int N = A.size() * B.size();
    const int M = 2;
    matrix_type Grid(N, M);
    matrix_type col1 = B.transpose().replicate(A.size(), 1);
    Grid.col(0) = A.replicate(B.size(), 1);
    Grid.col(1) = Eigen::Map<vector_type>(col1.data(), N, 1);
    return Grid;
  };


  
  template< typename RealType = int >
  Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>
  expand_grid(
    const Eigen::Matrix<RealType, Eigen::Dynamic, 1>& A,
    const Eigen::Matrix<RealType, Eigen::Dynamic, 1>& B,
    const Eigen::Matrix<RealType, Eigen::Dynamic, 1>& C
  ) {
    typedef typename
      Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>
      matrix_type;
    typedef typename Eigen::Matrix<RealType, Eigen::Dynamic, 1>
      vector_type;
    const int N = A.size() * B.size() * C.size();
    const int M = 3;
    matrix_type Grid(N, M);
    matrix_type col1 = B.transpose().replicate(A.size(), C.size());
    matrix_type col2 =
      C.transpose().replicate(A.size() * B.size(), 1);
    Grid.col(0) = A.replicate(B.size() * C.size(), 1);
    Grid.col(1) = Eigen::Map<vector_type>(col1.data(), N, 1);
    Grid.col(2) = Eigen::Map<vector_type>(col2.data(), N, 1);
    return Grid;
  };
  


  Eigen::MatrixXi neighborhood_cube_matrix_3d(
    const Eigen::Vector3i& half_dims
  ) {
    const Eigen::Vector3i D =
      (2 * half_dims.array() + 1).matrix();
    Eigen::VectorXi A = Eigen::VectorXi::LinSpaced(
      D[0], -half_dims[0], half_dims[0]);
    Eigen::VectorXi B = Eigen::VectorXi::LinSpaced(
      D[1], -half_dims[1], half_dims[1]);
    Eigen::VectorXi C = Eigen::VectorXi::LinSpaced(
      D[2], -half_dims[2], half_dims[2]);
    return stickygpm::expand_grid<>(A, B, C);
  };

  



  template< typename RealType >
  neighborhood_ball<RealType> neighborhood_perturbation(
    const stickygpm::qform_type& Qform,  // 
    const RealType& radius               // in mm units
  ) {
    if (radius < 0) {
      throw std::domain_error(
        "neighborhood_perturbation: radius cannot be negative");
    }
    const float radf = (float)radius;
    const Eigen::Vector3f voxel_dims =
      stickygpm::voxel_dimensions(Qform);
    const Eigen::Vector3i nbr_range =
      (radf / voxel_dims.array()).cast<int>().matrix();
    neighborhood_ball<RealType> Ball;
    Eigen::MatrixXi P =
      stickygpm::neighborhood_cube_matrix_3d(nbr_range);
    Eigen::MatrixXf Q = P.cast<float>();
    Q.conservativeResize(P.rows(), P.cols() + 1);
    Q.col(Q.cols() - 1) = Eigen::VectorXf::Zero(Q.rows(), 1);
    Q = (Q * Qform.transpose()).eval();
    Q.conservativeResize(Q.rows(), Q.cols() - 1);
    // std::vector<int> ball_indices(P.rows());
    std::vector<int> ball_indices;
    std::vector<int> all_cols{0, 1, 2};
    int within_radius_count = 0;
    ball_indices.reserve(P.rows());
    for (int i = 0; i < P.rows(); i++) {
      if (Q.row(i).norm() <= radf) {
	ball_indices.push_back(i);
	within_radius_count++;
      }
    }
    ball_indices.shrink_to_fit();
    if ( 1.0 - (double)within_radius_count / P.rows() > 0.55 ) {
      // Sphere within a cube: volume of sphere is about 52.4%
      //   so > 0.55 is a very rough check
      std::cout << "Warning: neighborhood_perturbation: "
		<< (1.0 - (double)within_radius_count / P.rows()) * 100
		<< "% of perturbations outside radius "
		<< radius << std::endl;
      if (within_radius_count == 0)
	throw std::domain_error(
          "neighborhood_perturbation: no perturbations within radius");
    }
    Ball.perturbations = Eigen::MatrixXi(within_radius_count, 3);
    Ball.distances = Eigen::Matrix<RealType, Eigen::Dynamic, 1>
      (within_radius_count);
    for (int i = 0; i < within_radius_count; i++) {
      Ball.perturbations.row(i) = P.row(ball_indices[i]);
      Ball.distances[i] = (RealType)Q.row(ball_indices[i]).norm();      
    }
    return Ball;
  };



  template< typename RealType >
  int neighborhood_cardinality(
    const ::nifti_image* const nii,
    const RealType radius
  ) {
    stickygpm::neighborhood_ball<RealType> ball =
      stickygpm::neighborhood_perturbation(
        stickygpm::qform_matrix(nii),
	radius
      );
    return ball.distances.size();
  };



};


#endif  // _STICKYGPM_VOXEL_NEIGHBORHOODS_

