
#include <algorithm>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <memory>
#include <nlopt.hpp>
#include <random>
#include <stdexcept>
#include <vector>


#include <chrono>
#include <iostream>

#include "stickygpm/covariance_functors.h"
#include "stickygpm/eigen_slicing.h"
#include "stickygpm/knots.h"
#include "stickygpm/utilities.h"
#include "stickygpm/stickygpm_regression_data.h"



#ifndef _STICKYGPM_ESTIMATE_KERNEL_PARAMETERS_
#define _STICKYGPM_ESTIMATE_KERNEL_PARAMETERS_

namespace stickygpm {


  template< typename T >
  struct sgpreg_data_summary {
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Y;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> locations;
    // Eigen::Matrix<T, Eigen::Dynamic, 1> sill;
    Eigen::Matrix<T, Eigen::Dynamic, 1> xtx;
    std::shared_ptr< stickygpm::covariance_functor<T> > cov_ptr;
  };



  template< typename T >
  stickygpm::sgpreg_data_summary<T>
  extract_data_summary(
    const stickygpm::stickygpm_regression_data<T>& data,
    const std::shared_ptr< stickygpm::covariance_functor<T> > cov_ptr,
    const int nloc = 2048
  ) {
    // typedef Eigen::Matrix<T, Eigen::Dynamic, 1> vector_type;
    stickygpm::sgpreg_data_summary<T> summ;
    std::vector<int> subset( data.Y().rows() );
    std::iota( subset.begin(), subset.end(), 0 );
    std::shuffle( subset.begin(), subset.end(), stickygpm::rng() );
    const Eigen::VectorXi subset_rows_ = Eigen::Map<Eigen::VectorXi>
      ( subset.data(), std::min( (int)subset.size(), nloc ) );
    const Eigen::VectorXi all_cols_ =
      Eigen::VectorXi::LinSpaced( data.n(), 0, data.n() - 1 );
    const Eigen::VectorXi all_cols_knots_ = Eigen::VectorXi::LinSpaced(
      data.mask_locations().cols(), 0, data.mask_locations().cols() - 1 );
    summ.Y = stickygpm::nullary_index( data.Y(), subset_rows_, all_cols_ );
    summ.locations = stickygpm::nullary_index(
      data.mask_locations(),
      subset_rows_,
      all_cols_knots_
    );
    //
    // summ.sill = summ.Y.rowwise().squaredNorm() / summ.Y.cols();
    //
    summ.xtx = data.X().rowwise().squaredNorm();
    summ.cov_ptr = cov_ptr;
    return summ;
  };
  
  
  

  template< typename T >
  double sgpreg_log_marginal_likelihood(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& Y,
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& locations,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& xtx,
    const std::shared_ptr< stickygpm::covariance_functor<T> > cov_ptr,
    const T nugget
  ) {
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> matrix_type;
    static int calls = 0;
      
    // auto start_t = std::chrono::high_resolution_clock::now();
    const Eigen::LLT<matrix_type> llt_of_K =
      stickygpm::knot_covariance_matrix( locations, *cov_ptr, nugget )
      .llt();
    const double ata = llt_of_K.solve( llt_of_K.matrixL() * Y )
      .colwise().squaredNorm().template cast<double>().sum();
    double log_det_K = 0, temp, lml;
    for ( int i = 0; i < locations.rows(); i++ ) {
      temp = (double)llt_of_K.matrixL().coeff(i, i);
      // ^^ NB: TriangularViewImpl::coeffRef is non-const
      log_det_K += std::log( temp * temp );
    }

    lml = -0.5 *
      ( ata +
	xtx.size() * log_det_K +
	locations.rows() * xtx.template cast<double>()
	  .array().log().sum()
	);
    // auto stop_t = std::chrono::high_resolution_clock::now();
    // auto diff_t = std::chrono::duration_cast<std::chrono::microseconds>
    //   ( stop_t - start_t );
    // std::cout << "\nComputation took " << ((double)diff_t.count() / 1e6)
    // 	      << " sec" << std::endl;

    // std::cout << "[" << calls << "]\t" << lml << std::endl;
    std::string msg = std::string("[") + std::to_string(calls) +
      std::string("]  ") + std::to_string(lml);
    if ( msg.size() < 50 ) {
      msg += std::string( 50 - msg.size(), ' ' );
    }
    else if ( msg.size() > 50 ) {
      msg = msg.substr( 0, 49 );
    }
    std::cout << std::string( 50, '\b' )
	      << msg << std::flush;
    calls++;
    
    return lml;
  };




  


  template< typename T >
  double snlml_nlopt(
    const std::vector<double> &x,
    std::vector<double>& grad,
    void* data_
  ) {
    typedef typename stickygpm::covariance_functor<T>::param_type
      param_type;
    stickygpm::sgpreg_data_summary<T>* data =
      (stickygpm::sgpreg_data_summary<T>*)data_;
    std::vector<T> theta( x.size() - 1 );
    for ( int i = 0; i < (int)x.size() - 1; i++ ) {
      theta[i] = static_cast<T>( x[i] );
    }
    param_type par( theta.cbegin(), theta.cend() );
    data->cov_ptr->param( par );
    return -0.5 * static_cast<double>(
      stickygpm::sgpreg_log_marginal_likelihood(
        data->Y,
	data->locations,
	data->xtx,
	data->cov_ptr,
	static_cast<T>( x.back() )
      ) );
  };

  



  
  template< typename T >
  int estimate_covariance_parameters(
    const std::shared_ptr< stickygpm::covariance_functor<T> > cov_ptr,
    const stickygpm::stickygpm_regression_data<T>& data,
    const int nloc = 2048,
    const nlopt::algorithm alg = nlopt::LN_NEWUOA_BOUND,
    const double xtol = 1e-5,
    const double huge_val = 100,
    const double eps0 = 1e-5
  ) {
    typedef typename stickygpm::covariance_functor<T>::param_type
      param_type;
    double min_obj;
    int code = 0;
    stickygpm::sgpreg_data_summary<T> summ =
      stickygpm::extract_data_summary( data, cov_ptr, nloc );
    param_type par = cov_ptr->param();
    std::vector<double> x( par.size() + 1 );
    for ( int i = 0; i < (int)par.size(); i++ ) {
      x[i] = static_cast<double>( par[i] );
    }
    // x[ x.size() - 1 ] = 2 * x[0];
    // x = { Cov Params, nugget variance }
    std::vector<double> lb( x.size(), eps0 );
    std::vector<double> ub( x.size(), huge_val );
    for ( int i = 0; i < (int)par.size(); i++ ) {
      lb[i] = static_cast<double>(cov_ptr->param_lower_bounds()[i]) +
	eps0;
      ub[i] = static_cast<double>(cov_ptr->param_upper_bounds()[i]) -
	eps0;
      ub[i] = (ub[i] > huge_val) ? huge_val : ub[i];
      if ( x[i] >= ub[i] ) {
	x[i] = (ub[i] - lb[i]) / 2 + lb[i];
      }
    }
    x.back() = ( 2 * x[0] >= ub.back() ) ?
      (ub.back() - lb.back()) / 2 + lb.back() : 2 * x[0];
    nlopt::opt optimizer( alg, x.size() );
    optimizer.set_lower_bounds( lb );
    optimizer.set_upper_bounds( ub );
    optimizer.set_min_objective( stickygpm::snlml_nlopt<T>, &summ );
    optimizer.set_xtol_rel( xtol );
    try {
      optimizer.optimize( x, min_obj );
      for ( int i = 0; i < (int)par.size(); i++ ) {
	par[i] = static_cast<T>( x[i] );
      }
      std::cout << "\n(Nugget variance = " << x.back() << ")"
		<< std::endl;
      cov_ptr->param( par );
    }
    catch ( std::exception &err ) {
      code = 1;
      std::cout << "\nError occurred:\n"
		<< err.what()
		<< std::endl;
    }
    return code;
  };







  
  
};

#endif  // _STICKYGPM_ESTIMATE_KERNEL_PARAMETERS_









  // // Variant with sill varinace
  // //
  // // -----------------------------------------------------------------

  // template< typename T >
  // double sgpreg_log_marginal_likelihood(
  //   const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& Y,
  //   const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& locations,
  //   const Eigen::Matrix<T, Eigen::Dynamic, 1>& xtx,
  //   const std::shared_ptr< stickygpm::covariance_functor<T> > cov_ptr,
  //   const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& sill
  // ) {
  //   typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> matrix_type;
  //   static int calls = 0;
    
  //   const Eigen::LLT<matrix_type> llt_of_K =
  //     stickygpm::knot_covariance_matrix( locations, *cov_ptr, sill )
  //     .llt();
  //   const double ata = llt_of_K.solve( llt_of_K.matrixL() * Y )
  //     .colwise().squaredNorm().template cast<double>().sum();
  //   double log_det_K = 0, temp, lml;
  //   for ( int i = 0; i < locations.rows(); i++ ) {
  //     temp = (double)llt_of_K.matrixL().coeff(i, i);
  //     // ^^ NB: TriangularViewImpl::coeffRef is non-const
  //     log_det_K += std::log( temp * temp );
  //   }

  //   lml = -0.5 *
  //     ( ata +
  // 	xtx.size() * log_det_K +
  // 	locations.rows() * xtx.template cast<double>()
  // 	.array().log().sum()
  // 	);
    
  //   std::string msg = std::string("[") + std::to_string(calls) +
  //     std::string("]  ") + std::to_string(lml);
  //   if ( msg.size() < 50 ) {
  //     msg += std::string( 50 - msg.size(), ' ' );
  //   }
  //   else if ( msg.size() > 50 ) {
  //     msg = msg.substr( 0, 49 );
  //   }
  //   std::cout << std::string( 50, '\b' )
  // 	      << msg << std::flush;
  //   calls++;
    
  //   return lml;
  // };


  


  // template< typename T >
  // double snlml_nlopt2(
  //   const std::vector<double> &x,
  //   std::vector<double>& grad,
  //   void* data_
  // ) {
  //   typedef typename stickygpm::covariance_functor<T>::param_type
  //     param_type;
  //   stickygpm::sgpreg_data_summary<T>* data =
  //     (stickygpm::sgpreg_data_summary<T>*)data_;
  //   std::vector<T> theta( x.size() - 1 );
  //   for ( int i = 0; i < x.size() - 1; i++ ) {
  //     theta[i] = static_cast<T>( x[i] );
  //   }
  //   param_type par( theta.cbegin(), theta.cend() );
  //   data->cov_ptr->param( par );
  //   return -0.5 * static_cast<double>(
  //     stickygpm::sgpreg_log_marginal_likelihood(
  //       data->Y,
  // 	data->locations,
  // 	data->xtx,
  // 	data->cov_ptr,
  // 	data->sill
  //     ) );
  // };

  



  
  // template< typename T >
  // int estimate_covariance_parameters2(
  //   const std::shared_ptr< stickygpm::covariance_functor<T> > cov_ptr,
  //   const stickygpm::stickygpm_regression_data<T>& data,
  //   const int nloc = 2048,
  //   const nlopt::algorithm alg = nlopt::LN_NEWUOA_BOUND,
  //   const double xtol = 1e-5,
  //   const double huge_val = 100,
  //   const double eps0 = 1e-5
  // ) {
  //   typedef typename stickygpm::covariance_functor<T>::param_type
  //     param_type;
  //   double min_obj;
  //   int code = 0;
  //   stickygpm::sgpreg_data_summary<T> summ =
  //     stickygpm::extract_data_summary( data, cov_ptr, nloc );
  //   param_type par = cov_ptr->param();
  //   std::vector<double> x( par.size() );
  //   for ( int i = 0; i < par.size(); i++ ) {
  //     x[i] = static_cast<double>( par[i] );
  //   }
  //   // x[ x.size() - 1 ] = 2 * x[0];
  //   // x = { Cov Params, nugget variance }
  //   std::vector<double> lb( x.size(), eps0 );
  //   std::vector<double> ub( x.size(), huge_val );
  //   for ( int i = 0; i < par.size(); i++ ) {
  //     lb[i] = static_cast<double>(cov_ptr->param_lower_bounds()[i]) +
  // 	eps0;
  //     ub[i] = static_cast<double>(cov_ptr->param_upper_bounds()[i]) -
  // 	eps0;
  //     ub[i] = (ub[i] > huge_val) ? huge_val : ub[i];
  //     if ( x[i] >= ub[i] ) {
  // 	x[i] = (ub[i] - lb[i]) / 2 + lb[i];
  //     }
  //   }
  //   // x.back() = ( 2 * x[0] >= ub.back() ) ?
  //   //   (ub.back() - lb.back()) / 2 + lb.back() : 2 * x[0];
  //   nlopt::opt optimizer( alg, x.size() );
  //   optimizer.set_lower_bounds( lb );
  //   optimizer.set_upper_bounds( ub );
  //   optimizer.set_min_objective( stickygpm::snlml_nlopt<T>, &summ );
  //   optimizer.set_xtol_rel( xtol );
  //   try {
  //     optimizer.optimize( x, min_obj );
  //     for ( int i = 0; i < par.size(); i++ ) {
  // 	par[i] = static_cast<T>( x[i] );
  //     }
  //     // std::cout << "\n(Nugget variance = " << x.back()] << ")"
  //     // 		<< std::endl;
  //     cov_ptr->param( par );
  //   }
  //   catch ( std::exception &err ) {
  //     code = 1;
  //     std::cout << "\nError occurred:\n"
  // 		<< err.what()
  // 		<< std::endl;
  //   }
  //   return code;
  // };


// -------------------------------------------------------------------







  // template< typename T >
  // double sgpreg_log_marginal_likelihood2(
  //   const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& Y,
  //   const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& locations,
  //   const Eigen::Matrix<T, Eigen::Dynamic, 1>& xtx,
  //   const std::shared_ptr< stickygpm::covariance_functor<T> > cov_ptr,
  //   const T nugget,
  //   std::vector<double>& grad
  // ) {
  //   typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> matrix_type;
  //   typedef Eigen::Matrix<T, Eigen::Dynamic, 1> vector_type;
  //    
  //   // auto start_t = std::chrono::high_resolution_clock::now();
  //   const int n = Y.cols();
  //   const int nloc = Y.rows();
  //   const Eigen::LLT<matrix_type> llt_of_K =
  //     stickygpm::knot_covariance_matrix( locations, *cov_ptr, nugget )
  //     .llt();
  //   const double ata = llt_of_K.solve( llt_of_K.matrixL() * Y )
  //     .colwise().squaredNorm().template cast<double>().sum();
  //   vector_type alpha;
  //   matrix_type Kinv, dK;
  //   double lml, temp;
  //   double log_det_K = 0;
  //   for ( int l = 0; l < nloc; l++ ) {
  //     temp = (double)llt_of_K.matrixL().coeff(l, l);
  //     // ^^ NB: TriangularViewImpl::coeffRef is non-const
  //     log_det_K += std::log( temp * temp );
  //   }
  //
  //   if ( !grad.empty() ) {
  //     std::cout << "Grad: (" << std::flush;
  //     Kinv = llt_of_K.solve( matrix_type::Identity(nloc, nloc) );
  //     for ( int j = 0; j < (int)grad.size(); j++ ) {
  // 	grad[j] = 0;
  // 	if ( j < (int)cov_ptr->param_size() ) {
  // 	  dK = stickygpm
  // 	    ::covariance_matrix_gradient( locations, *cov_ptr, j );
  // 	}
  // 	for ( int i = 0; i < n; i++ ) {
  // 	  alpha = Kinv * Y.col(i);
  // 	  if ( j < (int)cov_ptr->param_size() ) {
  // 	    for ( int l = 0; l < nloc; l++ ) {
  // 	      grad[j] += 0.5 * static_cast<double>(
  //               (( alpha.coeffRef(l) * alpha.transpose() -
  // 		   Kinv.row(l) ) * dK.col(l) ).coeff(0)
  //             );
  // 	    }
  // 	  }
  // 	  else {
  // 	    for ( int l = 0; l < nloc; l++ ) {
  // 	      grad[j] += 0.5 * static_cast<double>(
  //               alpha.coeffRef(l) * alpha.coeffRef(l) - Kinv.coeffRef(l, l)
  //             );
  // 	    }
  // 	  }
  // 	   // end - if ( j < (int)cov_ptr->param_size() ) : else
  // 	}  // for ( int i = 0; i < n; i++ )
  // 	std::cout << -grad[j] << ", ";
  //     }    // for ( int j = 0; j < (int)grad.size(); j++ )
  //     std::cout << "\b\b)'" << std::endl;
  //    
  //   }
  //   // if ( !grad.empty() )
  //
  //   lml = -0.5 *
  //     ( ata +
  // 	xtx.size() * log_det_K +
  // 	locations.rows() * xtx.template cast<double>()
  // 	.array().log().sum()
  // 	);
  //   // auto stop_t = std::chrono::high_resolution_clock::now();
  //   // auto diff_t = std::chrono::duration_cast<std::chrono::microseconds>
  //   //   ( stop_t - start_t );
  //   // std::cout << "\nComputation took " << ((double)diff_t.count() / 1e6)
  //   // 	      << " sec" << std::endl;
  //   std::cout << lml << ": " << ata << ", " << log_det_K << std::endl;
  //  
  //   return lml;
  // };







