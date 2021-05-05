
#include <algorithm>
#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <memory>
#include <nifti1.h>
#include <nifti1_io.h>
#include <random>
#include <string>
#include <vector>

#include "stickygpm/covariance_functors.h"
#include "stickygpm/extra_distributions.h"
#include "stickygpm/utilities.h"
#include "stickygpm/nifti_manipulation.h"
#include "stickygpm/outer_rlsbp2.h"                // <- *** 2
#include "stickygpm/projected_gp_regression2.h"    // <- *** 2
#include "stickygpm/stickygpm_regression_data.h"
#include "stickygpm/vector_summation.h"



#ifndef _STICKYGPM_REGRESSION_MODEL_
#define _STICKYGPM_REGRESSION_MODEL_

namespace stickygpm {
  
  template< typename T >
  class stickygpm_regression_model {
  public:
    typedef T scalar_type;
    typedef typename
    Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>
    matrix_type;
    typedef typename Eigen::Matrix<scalar_type, Eigen::Dynamic, 1>
    vector_type;

    //
    class output {
    public:
      output() { ; }
      output(
	const std::string basename,
        const int nvox,
	const int lsbp_trunc,
	const bool output_samples
      );

      void update(
        const stickygpm::stickygpm_regression_data<T>& data,
	const outer_rlsbp< stickygpm::projected_gp_regression<T> >& lsbp,
	const vector_type& sigma_sq_inv,
	const T log_likelihood,
	const T shape_sigma,
	const T rate_sigma
      );
      void close_logs();

      bool write_posterior_summaries(
        ::nifti_image* const mask,
	const std::string basename,
	const matrix_type& knot_locations
      ) const;


      double deviance(
        const stickygpm::stickygpm_regression_data<T>& data
      ) const;
      
      double dic(
        const stickygpm::stickygpm_regression_data<T>& data
      ) const;
		  

    private:
      static const double _log_2pi;
      int _updates;
      bool _output_samples;
      double _loglik_first_moment;
      std::vector< std::ofstream > _beta_log;
      std::vector< std::ofstream > _lsbp_coeffs_log;
      // std::ofstream _sigma_log;
      std::ofstream _etc_log;  // log likelihood, sigma hyperparameter, cluster probs
      std::ofstream _cluster_log;
      std::vector< matrix_type > _beta_first_moment;
      std::vector< matrix_type > _beta_second_moment;
      std::vector< matrix_type > _lsbp_coeffs_second_moment;
      matrix_type _cluster_assignment_counts;
      matrix_type _lsbp_coeffs_first_moment;
      vector_type _sigma_sq_first_moment;
    };
    // end - class output
    
    
    stickygpm_regression_model(
      const stickygpm::stickygpm_regression_data<T>& data,
      const ::nifti_image* mask,
      const std::shared_ptr< stickygpm::covariance_functor<T> > cov_ptr,
      const std::string output_basename,
      const bool output_samples = false,
      const int lsbp_truncation = 10,
      const T lsbp_mu = 0,
      const T lsbp_sigma = 1,
      const int nknots = 2000,
      const int basis_neighborhood = 2000,
      const int init_kmeans = 1,
      const double repulsion = 0
    );

    void run_mcmc(
      const stickygpm::stickygpm_regression_data<T>& data,
      const int burnin,
      const int nsave,
      const int thin,
      const bool output_samples,
      const bool verbose
    );

    void close_logs();

    bool write_posterior_summaries( ::nifti_image* const mask ) const;
    bool write_deviance_summary(
      const stickygpm::stickygpm_regression_data<T>& data
    ) const;
    

  private:
    std::shared_ptr<
    typename stickygpm::projected_gp_regression<T>::shared_data >
    _gp_basis_ptr_;

    outer_rlsbp< stickygpm::projected_gp_regression<T> >
    _lsbp_;

    vector_type _sigma_sq_inv;
    vector_type _squared_residuals;
    vector_type _cluster_occupancy;

    scalar_type _shape_sigma;
    scalar_type _rate_sigma;
    scalar_type _a;  // hyperparameter (shape) for _rate_sigma
    scalar_type _b;  // hyperparameter (rate) for _rate_sigma

    double _avg_cluster_separation;
    double _avg_acceptance_rate;

    std::string _output_base;

    output _output_;

    void _update_sigma(
      const stickygpm::stickygpm_regression_data<T>& data
    );
    void _update_sigma_hyperparameters();
  };
  
};




// --- stickygpm_regression_model<T>::output -------------------------


template< typename T >
const double stickygpm::stickygpm_regression_model<T>::output
::_log_2pi = 1.837877066409345339082;


template< typename T >
stickygpm::stickygpm_regression_model<T>::output::output(
  const std::string basename,
  const int nvox,
  const int lsbp_trunc,
  const bool output_samples
) {
  // const std::string sigma_log_file = basename +
  //   std::string( "_sigma.dat" );
  const std::string etc_log_file = basename +
    std::string( "_etc.dat" );
  const std::string cluster_log_file = basename +
    std::string( "_cluster.dat" );
  const std::string beta_log_base = basename +
    std::string( "_beta_star_cluster" );
  const std::string lsbp_log_base = basename +
    std::string( "_lsbp_coeffs" );
  //
  _updates = 0;
  _output_samples = output_samples;
  _loglik_first_moment = 0;
  _beta_first_moment.reserve( lsbp_trunc );
  _beta_second_moment.reserve( lsbp_trunc );
  _lsbp_coeffs_second_moment.reserve( lsbp_trunc );
  _sigma_sq_first_moment = vector_type( nvox );
  //
  if ( output_samples ) {
    // _sigma_log = std::ofstream( sigma_log_file );
    _etc_log = std::ofstream( etc_log_file );
    _cluster_log = std::ofstream( cluster_log_file );
    _beta_log.reserve( lsbp_trunc );
    _lsbp_coeffs_log.reserve( lsbp_trunc );
    for ( int k = 0; k < lsbp_trunc; k++ ) {
      std::string beta_log_file = beta_log_base +
	std::to_string( k ) + std::string(".dat");
      std::string lsbp_log_file = lsbp_log_base +
	std::to_string( k ) + std::string(".dat");
      //
      _beta_log.push_back( std::ofstream(beta_log_file) );
      _lsbp_coeffs_log.push_back( std::ofstream(lsbp_log_file) );
    }
  }
};




template< typename T >
void stickygpm::stickygpm_regression_model<T>::output::update(
  const stickygpm::stickygpm_regression_data<T>& data,
  const outer_rlsbp< stickygpm::projected_gp_regression<T> >& lsbp,
  const vector_type& sigma_sq_inv,
  const T log_likelihood,
  const T shape_sigma,
  const T rate_sigma
) {
  // Initialize storage if this is the first update
  if ( _updates == 0 ) {
    for ( int k = 0; k < lsbp.truncation(); k++ ) {
      _beta_first_moment.push_back(
        matrix_type::Zero( data.Y().rows(), data.X().cols() )
      );
      _beta_second_moment.push_back(
        matrix_type::Zero( data.Y().rows(), data.X().cols() )
      );
      _lsbp_coeffs_second_moment.push_back(
        matrix_type::Zero( data.Z().cols(), data.Z().cols() )
      );
    }
    _sigma_sq_first_moment.setZero( data.Y().rows() );
    _lsbp_coeffs_first_moment.setZero(
      lsbp.logistic_coefficients().rows(),
      lsbp.logistic_coefficients().cols()
    );

    _cluster_assignment_counts =
      matrix_type::Zero( data.n(), lsbp.truncation() );

    if ( _output_samples ) {
      // Add variable names to _etc_log
      _etc_log << "LogLikelihood\tShapeSigma\tRateSigma";
      for ( int k = 0; k < lsbp.truncation(); k++ ) {
	_etc_log << "\tPr(k=" << k << ")";
      }
      _etc_log << std::endl;
    }
  }
  // end - Initialize storage

  // Update stored summaries
  vector_type proj_beta;
  for ( int k = 0; k < lsbp.truncation(); k++ ) {
    for ( int j = 0; j < data.X().cols(); j++ ) {
      proj_beta = lsbp.inner_model_ref( k ).projected_beta( j );
      _beta_first_moment[k].col( j ) += proj_beta;
      _beta_second_moment[k].col( j ) += proj_beta.cwiseAbs2();
    }
    _lsbp_coeffs_second_moment[ k ] +=
      lsbp.logistic_coefficients().col( k ) *
      lsbp.logistic_coefficients().col( k ).transpose();
  }
  _sigma_sq_first_moment += sigma_sq_inv.cwiseInverse();
  _lsbp_coeffs_first_moment += lsbp.logistic_coefficients();
  _loglik_first_moment += log_likelihood;

  for ( int i = 0; i < data.n(); i++ ) {
    _cluster_assignment_counts.coeffRef( i, lsbp.cluster_label(i) )++;
  }
  // end - Update sotred summaries

  // Update log files if requested
  if ( _output_samples ) {

    for ( int k = 0; k < lsbp.truncation(); k++ ) {
      _beta_log[k] << lsbp.inner_model_ref( k ).parameter_vector().transpose()
		   << std::endl;
      _lsbp_coeffs_log[k] <<
	lsbp.logistic_coefficients().col( k ).transpose()
			  << std::endl;
    }
    _etc_log << log_likelihood << "\t"
	     << shape_sigma << "\t"
	     << rate_sigma << "\t"
	     << lsbp.realized_cluster_probability().transpose()
	     << std::endl;

    for ( int i = 0; i < data.n(); i++ ) {
      _cluster_log << lsbp.cluster_label( i );
      if ( i < (data.n() - 1) ) {
	_cluster_log << "\t";
      }
    }
    _cluster_log << std::endl;
    
  }
  // if ( _output_samples )
  
  //
  _updates++;
  //
};





template< typename T >
bool stickygpm::stickygpm_regression_model<T>::output
::write_posterior_summaries(
  ::nifti_image* const mask,
  const std::string basename,
  const typename stickygpm::stickygpm_regression_model<T>
    ::matrix_type& knot_locations
) const {
  const std::string sep = "\n\t";
  std::string failed = "";
  std::string lsbp_file;
  std::ofstream lsbp_log;
  vector_type temp;

  // Loop over clustered parameters
  for ( int k = 0; k < (int)_beta_first_moment.size(); k++ ) {
    std::string beta_file_base = basename +
      std::string("_cluster") + std::to_string( k ) +
      std::string("_beta");
    std::string lsbp_file_base = basename +
      std::string("_cluster") + std::to_string( k ) +
      std::string("_lsbp_coefficients");
    
    // Write beta means
    mask->intent_code = NIFTI_INTENT_ESTIMATE;
    for ( int j = 0; j < _beta_first_moment[k].cols(); j++ ) {
      
      std::string beta_file = beta_file_base +
	std::to_string( j ) + "_posterior_mean.nii.gz";
      temp = _beta_first_moment[ k ].col( j ) / _updates;
      try {
	stickygpm::emplace_nonzero_data( mask, temp );
	stickygpm::nifti_image_write( mask, beta_file );
	std::cout << beta_file << " written!\n";
      }
      catch (...) {
	failed = failed + sep + beta_file;
      }

      // Write beta variances
      beta_file = beta_file_base +
	std::to_string( j ) + "_posterior_variance.nii.gz";
      temp = _beta_second_moment[ k ].col( j ) / _updates -
	temp.cwiseAbs2();
      try {
	stickygpm::emplace_nonzero_data( mask, temp );
	stickygpm::nifti_image_write( mask, beta_file );
	std::cout << beta_file << " written!\n";
      }
      catch (...) {
	failed = failed + sep + beta_file;
      }
    }
    // for ( int j = 0; j < _beta_first_moment[k].cols(); j++ )


    // Write LSBP coefficients' covariance
    lsbp_file = lsbp_file_base +
      std::string("_posterior_covariance.dat");
    lsbp_log.open( lsbp_file.c_str() );
    if ( lsbp_log ) {
      lsbp_log <<
	( _lsbp_coeffs_second_moment[k] / _updates -
	  _lsbp_coeffs_first_moment.col( k ) * 
	  _lsbp_coeffs_first_moment.col( k ).transpose() /
	  ( _updates * _updates ) )
	       << std::endl;
      lsbp_log.close();
      std::cout << lsbp_file << " written!\n";
    }
    else {
      failed = failed + sep + lsbp_file;
    }
    
  }
  // for ( int k = 0; k < _beta_first_moment.size(); k++ )


  // Write LSBP coefficients
  lsbp_file = basename +
      std::string("_lsbp_coefficients_posterior_mean.dat");
  lsbp_log.open( lsbp_file.c_str() );
  if ( lsbp_log ) {
    lsbp_log << ( _lsbp_coeffs_first_moment / _updates )
	     << std::endl;
    lsbp_log.close();
    std::cout << lsbp_file << " written!\n";
  }
  else {
    failed = failed + sep + lsbp_file;
  }

  // Write cluster allotment
  lsbp_file = basename + std::string("_cluster_probability.dat");
  lsbp_log.open( lsbp_file.c_str() );
  if ( lsbp_log ) {
    lsbp_log << ( _cluster_assignment_counts / _updates )
	     << std::endl;
    lsbp_log.close();
    std::cout << lsbp_file << " written!\n";
  }
  else {
    failed = failed + sep + lsbp_file;
  }
  
  
  // Write sigma_sq mean
  std::string sigma_file = basename +
    std::string("_sigma_sq_posterior_mean.nii.gz");
  temp = _sigma_sq_first_moment / _updates;
  try {
    stickygpm::emplace_nonzero_data( mask, temp );
    stickygpm::nifti_image_write( mask, sigma_file );
    std::cout << sigma_file << " written!" << std::endl;
  }
  catch (...) {
    failed = failed + sep + sigma_file;
  }


  // Write knot locations
  std::string knot_file = basename + std::string("_knots.nii.gz");
  try {
    ::nifti_image* knot_img =
      stickygpm::nifti_image_read( sigma_file, 1 );
    stickygpm::make_knot_image( knot_img, knot_locations );
    stickygpm::nifti_image_write( knot_img, knot_file );
    ::nifti_image_free( knot_img );
    std::cout << knot_file << " written!" << std::endl;
  }
  catch (...) {
    failed = failed + sep + knot_file;
  }

  // If any writes failed, print warning:
  if ( !failed.empty() ) {
    std::cerr << "*** The following files FAILED to write:\n"
	      << failed
	      << std::endl;
  }
  
  return failed.empty();
};






template< typename T >
double stickygpm::stickygpm_regression_model<T>::output
::deviance(
  const stickygpm::stickygpm_regression_data<T>& data
) const {
  const double norm_c = -0.5 * data.Y().size() * _log_2pi;
  return -2 * ( _loglik_first_moment / _updates + norm_c );
};



template< typename T >
double stickygpm::stickygpm_regression_model<T>::output::dic(
  const stickygpm::stickygpm_regression_data<T>& data
) const {
  // DIC_8 from:
  // -----------
  // Celeux, Gilles, et al. "Deviance information criteria for missing
  //   data models." Bayesian analysis 1.4 (2006): 651-673.
  //
  const vector_type sigma_inv =
    ( _sigma_sq_first_moment / _updates ).cwiseSqrt().cwiseInverse();
  const double sum_log_sigma_inv = sigma_inv.template cast<double>()
    .array().log().sum();
  const double norm_c = -0.5 * data.Y().size() * _log_2pi;
  vector_type residuals( data.Y().rows() );
  //
  double conditional_loglik = 0;
  double remaining_stick, p, pksum, cllkmax;
  Eigen::VectorXd pk( _lsbp_coeffs_first_moment.cols() );
  // ^^ Cluster probabilities
  Eigen::VectorXd cllk( pk.size() );  // Conditional log likelihood
  //
  std::vector< matrix_type > Beta( _beta_first_moment.size() );
  for ( int j = 0; j < (int)Beta.size(); j++ ) {
    Beta[j] = _beta_first_moment[j] / _updates;
  }
  //
  for ( int i = 0; i < data.n(); i++ ) {
    // Compute expected component weights for each subject
    pk = ( data.Z().row(i) * _lsbp_coeffs_first_moment )
      .transpose().template cast<double>() / _updates;
    remaining_stick = 1;
    pksum = 0;
    for ( int j = 0; j < pk.size(); j++ ) {
      pk.coeffRef(j) = 
	extra_distributions::std_logistic_cdf( pk.coeff(j) );
      p = remaining_stick * pk.coeff(j);
      remaining_stick *= ( 1 - pk.coeff(j) );
      pk.coeffRef(j) = p;
      //
      // Compute conditional (non-normalized) log likelihood
      residuals = sigma_inv.asDiagonal() *
	( data.Y().col(i) - Beta[j] * data.X().row(i).transpose()
	  );
      cllk.coeffRef(j) = -0.5 * stickygpm::vdot( residuals ) +
	sum_log_sigma_inv;
      cllkmax = ( j == 0 ) ? cllk.coeff(j) :
	std::max( cllkmax, cllk.coeff(j) );
      // conditional_loglik += pk.coeff(j) * llk;
    }
    for ( int j = 0; j < pk.size(); j++ ) {
      pk.coeffRef(j) *= std::exp( cllk.coeff(j) - cllkmax );
      pksum += pk.coeff(j);
    }
    for ( int j = 0; j < pk.size(); j++ ) {
      conditional_loglik += pk.coeff(j) * cllk.coeff(j) / pksum;
    }
  }
  //
  return 2 * deviance( data ) + 2 * ( conditional_loglik + norm_c );
};




  // const vector_type temp =
  //   sigma_sq_inv.cwiseSqrt().asDiagonal() *
  //   ( data.Y().col( i ) - _p_data->Bases() *
  //     ( _beta_star * data.X().row(i).transpose() )
  //     );
  // const double norm_c = 0.5 *
  //   ( sigma_sq_inv.array().template cast<double>().log().sum() -
  //     sigma_sq_inv.size() * std::log(2 * PI) );
  // return -0.5 * stickygpm::vdot( temp ) + norm_c;




template< typename T >
void stickygpm::stickygpm_regression_model<T>::output::close_logs() {
  if ( _output_samples ) {
    _etc_log.close();
    _cluster_log.close();
    for ( int k = 0; k < (int)_beta_log.size(); k++ ) {
      _beta_log[ k ].close();
      _lsbp_coeffs_log[ k ].close();
    }
  }
};








// --- stickygpm_regression_model<T> ---------------------------------


template< typename T >
stickygpm::stickygpm_regression_model<T>::stickygpm_regression_model(
  const stickygpm::stickygpm_regression_data<T>& data,
  const ::nifti_image* mask,
  const std::shared_ptr< stickygpm::covariance_functor<T> > cov_ptr,
  const std::string output_basename,
  const bool output_samples,
  const int lsbp_truncation,
  const T lsbp_mu,
  const T lsbp_sigma,
  const int nknots,
  const int basis_neighborhood,
  const int init_kmeans,
  const double repulsion
) {

  std::cout << "Computing GP bases" << std::endl;
  _gp_basis_ptr_ = std::make_shared< typename
      stickygpm::projected_gp_regression<scalar_type>::shared_data >
    ( mask, *cov_ptr, nknots, basis_neighborhood );

  if ( _gp_basis_ptr_->Sigma_llt().info() != Eigen::Success ) {
    std::cerr << "  *** ERROR: destined to fail (singular covariance)"
	      << std::endl;
  }

  std::cout << "Initializing LSBP {T = "
	    << lsbp_truncation << ", " << "\u03BC" << " = "
	    << lsbp_mu << ", " << "\u03C3" << " = "
	    << lsbp_sigma << "}"
	    << std::endl;
  _lsbp_ = outer_rlsbp< stickygpm::projected_gp_regression<T> >
    ( data, lsbp_truncation, lsbp_sigma, lsbp_mu, repulsion );
  // _lsbp_.repulsion_parameter( repulsion );

  // Initialize inner GP models and move them to _lsbp_
  std::vector< stickygpm::projected_gp_regression<scalar_type> >
    gpreg( lsbp_truncation );
  gpreg[0] = stickygpm::projected_gp_regression<scalar_type>
    ( _gp_basis_ptr_, data.X().cols() );
  for ( int k = 1; k < (int)gpreg.size(); k++ ) {
    gpreg[k] = gpreg[0];
  }
  // std::cout << "& Moving models " << std::endl;
  _lsbp_.move_models( gpreg.begin(), gpreg.end() );
  gpreg.clear();
  //

  _squared_residuals = vector_type::Zero( data.Y().rows() );
  _sigma_sq_inv =
    ( data.Y().rowwise().squaredNorm() / data.Y().cols() -
      data.Y().rowwise().mean().cwiseAbs2() )
    .cwiseInverse();
  _cluster_occupancy = vector_type::Zero( lsbp_truncation );
  //
  _shape_sigma = 1;
  _rate_sigma = 0;
  _a = 1;
  _b = 0;

  //
  _avg_cluster_separation = 0;
  _avg_acceptance_rate = 0.5;
  //

  std::cout << "sigma^2: ["
  	    << _sigma_sq_inv.cwiseInverse().minCoeff() << ", "
  	    << _sigma_sq_inv.cwiseInverse().mean() << ", "
  	    << _sigma_sq_inv.cwiseInverse().maxCoeff() << "]"
  	    << std::endl;

  std::cout << "Initializing output logs" << std::endl;
  _output_base = output_basename;
  _output_ = output(
    output_basename,
    data.Y().rows(),
    lsbp_truncation,
    output_samples
  );

  std::cout << "Computing warm start clustering" << std::endl;
  _lsbp_.initialize_clusters( data, init_kmeans );
  _lsbp_.print_cluster_sizes( std::cout );

  std::cout << "Model object complete!" << std::endl;
};





template< typename T >
void stickygpm::stickygpm_regression_model<T>::run_mcmc(
  const stickygpm::stickygpm_regression_data<T>& data,
  const int burnin,
  const int nsave,
  const int thin,
  const bool output_samples,
  const bool verbose
) {
  // const double pr_full = (data.n() <= 4) ? 1 : 1 - 4.0 / data.n();
  // const double pr_full = 0.5;
  const int half_burnin = burnin / 2;
  const double emaw = 0.02;  // Exponential moving average weight
  double pr_full = 0.8;
  stickygpm::utilities::progress_bar pb( nsave * thin + burnin );
  int save_count = 0;
  double loglik = 0, mhrate = 0;
  bool warmup_period, update_reflab, realign_labels;
  int iter = 0;
  std::cout << "Fitting model" << std::endl;
  while ( save_count != nsave ) {
    warmup_period = iter < burnin;
    update_reflab = iter < half_burnin  &&  iter >= (burnin / 10);
    realign_labels = iter >= half_burnin;
    if ( iter == half_burnin ) {
      _lsbp_.sort_clusters();
      pr_full = 0.9;
    }
    //
    mhrate = _lsbp_.update(
      data,
      _sigma_sq_inv,
      pr_full,
      update_reflab,
      realign_labels
    );
    _update_sigma( data );
    _update_sigma_hyperparameters();
    //
    // avgsep = 0.96 * avgsep + 0.04 * _lsbp_.min_cluster_distance();
    _avg_cluster_separation = (1 - emaw) * _avg_cluster_separation +
      emaw * _lsbp_.min_cluster_distance();
    //
    _avg_acceptance_rate = (1 - emaw) * _avg_acceptance_rate +
      emaw * mhrate;
    //
    if ( !warmup_period  &&  (iter % thin) == 0 ) {
      loglik = _lsbp_.log_likelihood();
      _output_.update(
        data, _lsbp_, _sigma_sq_inv,
	static_cast<scalar_type>( loglik ),
	_shape_sigma, _rate_sigma
      );
      _cluster_occupancy += _lsbp_.cluster_sizes();
      //
      save_count++;
    }
    iter++;

    // Print info
    if ( verbose ) {
      if ( warmup_period ) {
	loglik = _lsbp_.log_likelihood();
      }
      std::cout << "[" << iter << "]  LogPost: "
		<< loglik << " + " << _lsbp_.log_prior()
		<< "  (\u03B1 = " << mhrate << ")"
		<< "  ~  ";
      // << ( loglik + _lsbp_.log_prior() )
      _lsbp_.print_cluster_sizes( std::cout );
      std::cout << std::endl;
      //
      // std::cout << "beta_0: [";
      // for ( int j = 0; j < data.X().cols(); j++ ) {
      // 	std::cout << _lsbp_.inner_model_ref(0).projected_beta( j ).mean()
      // 		  << ", ";
      // }
      // std::cout << "\b\b] " << std::endl;
      //
    }
    else {
      pb++;
      std::cout << pb;
    }
  }
  // while ( int iter = 0, save_count != nsave )
  _cluster_occupancy /= save_count * data.n();
};




template< typename T >
void stickygpm::stickygpm_regression_model<T>::close_logs() {
  _output_.close_logs();
};



template< typename T >
bool stickygpm::stickygpm_regression_model<T>
::write_posterior_summaries(
  ::nifti_image* const mask
) const {
  const bool s =
    _output_.write_posterior_summaries(
      mask,
      _output_base,
      _gp_basis_ptr_->knots()
    );
  //
  std::cout << "\nCluster Occupancy:\n"
	    << "------------------\n";
  for ( int i = 0; i < _cluster_occupancy.size(); i++ ) {
    std::cout << "{#" << i << "}:  "
	      << (_cluster_occupancy[i] * 100)
	      << "%\n";
  }
  //
  return s;
};




template< typename T >
bool stickygpm::stickygpm_regression_model<T>
::write_deviance_summary(
  const stickygpm::stickygpm_regression_data<T>& data
) const {
  const double dev = _output_.deviance( data );
  const double dic = _output_.dic( data );
  const double pD  = dic - dev;
  const std::string fname = _output_base +
    std::string("_deviance_summary.csv");
  std::ofstream dev_summ( fname );
  // First write to stdout
  std::cout << "\nRepulsion parameter \u03C4:\t"
	    << _lsbp_.repulsion_parameter()
	    << "\nEffective Parameters:\t" << pD
	    << "\nDeviance:\t" << dev
	    << "\nDIC:\t\t" << dic
	    << "\n\nAvg. Cluster Separation:\t"
	    << _avg_cluster_separation
	    << "\nAvg. Rejection Rate:\t\t"
	    << (1 - _avg_acceptance_rate)
	    << "\n" << std::endl;
  //
  if ( dev_summ ) {
    dev_summ << "RepulsionParameter,Deviance,DIC,RejectionRate,"
	     << "AvgClusterSeparation\n";
    dev_summ << _lsbp_.repulsion_parameter() << ",";
    dev_summ << dev << ",";
    dev_summ << dic << ",";
    dev_summ << (1 - _avg_acceptance_rate) << ",";
    dev_summ << _avg_cluster_separation << "\n";
    dev_summ.close();
  }
  return dev_summ.good();
};





template< typename T >
void stickygpm::stickygpm_regression_model<T>::_update_sigma(
  const stickygpm::stickygpm_regression_data<T>& data
) {
  _squared_residuals.setZero( data.Y().rows() );
  for ( int i = 0; i < data.n(); i++ ) {
    _squared_residuals +=
      _lsbp_.residuals( data, i ).cwiseAbs2();
  }
  for ( int v = 0; v < data.Y().rows(); v++ ) {
    std::gamma_distribution<scalar_type> Gamma(
      0.5 * data.n() + _shape_sigma,
      1 / (0.5 * _squared_residuals.coeffRef(v) + _rate_sigma)
    );
    _sigma_sq_inv.coeffRef( v ) = Gamma( stickygpm::rng() );
  }
  // std::cout << "mean( sigma^2 ) = " << _sigma_sq_inv.cwiseInverse().mean()
  // 	    << ";  rate hyperparameter = " << _rate_sigma
  // 	    << std::endl;
};


template< typename T >
void stickygpm::stickygpm_regression_model<T>
::_update_sigma_hyperparameters() {
  std::gamma_distribution<scalar_type> Gamma(
    _sigma_sq_inv.size() * _shape_sigma + _a,
    1 / (_sigma_sq_inv.sum() + _b)
  );
  _rate_sigma = Gamma( stickygpm::rng() );
};


#endif  // _STICKYGPM_REGRESSION_MODEL_
