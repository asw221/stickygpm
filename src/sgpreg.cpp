
#include <iostream>
#include <memory>
#include <nifti1_io.h>
#include <omp.h>
#include <string>
#include <vector>

#include "stickygpm/covariance_functors.h"
#include "stickygpm/estimate_kernel_parameters.h"
#include "stickygpm/utilities.h"
#include "stickygpm/nifti_manipulation.h"
#include "stickygpm/sgpreg_command_parser.h"
// #include "stickygpm/outer_lsbp.h"
// #include "stickygpm/projected_gp_regression.h"
#include "stickygpm/stickygpm_regression_data.h"
#include "stickygpm/stickygpm_regression_model.h"







int main ( int argc, char* argv[] ) {
#ifdef STICKYGPM_DOUBLE_PRECISION
  typedef double scalar_type;
#else
  typedef float scalar_type;
#endif

  stickygpm::sgpreg_command_parser<scalar_type> inputs(argc, argv);
  if ( !inputs )
    return 1;
  else if ( inputs.help_invoked() )
    return 0;

  stickygpm::set_seed( inputs.seed() );
  stickygpm::set_monitor_simulations( inputs.monitor() );
  stickygpm::set_number_of_threads( inputs.threads() );
  ::omp_set_num_threads( stickygpm::threads() );
  Eigen::setNbThreads( stickygpm::threads() );


  // Printing of some inputs
  if ( !inputs.clustering_random_effects_files().empty() ) {
    std::cout << "LSBP Random effects files:";
    for ( std::string fn : inputs.clustering_random_effects_files() ) {
      std::cout << "\n\t" << fn;
    }
    std::cout << std::endl;
  }
  //

  ::nifti_image* mask =
      stickygpm::nifti_image_read( inputs.mask_file(), 1 );
  std::cout << "Read mask: " << inputs.mask_file() << std::endl;
  
  std::shared_ptr< stickygpm::covariance_functor<scalar_type> >
    _cov_ptr_;
  if ( inputs.radial_basis() ) {
    if ( inputs.covariance_parameters().empty() ) {
      // std::vector<scalar_type> theta_temp_{1, 1, 1};
      _cov_ptr_ =
        std::make_shared< stickygpm::radial_basis<scalar_type> >();
      // theta_temp_.clear();
    }
    else {
      _cov_ptr_ =
        std::make_shared< stickygpm::radial_basis<scalar_type> >(
          inputs.covariance_parameters().cbegin(),
	  inputs.covariance_parameters().cend()
        );
    }
  }
  else {
    std::cerr << "Unrecognized covaraince function type\n";
    return 1;
  }


  stickygpm::stickygpm_regression_data<scalar_type> _data_(
    inputs.data_files(),
    inputs.mask_file(),
    inputs.covariates_file(),
    inputs.lsbp_intercept()
  );
  if ( !inputs.clustering_covariates_file().empty() ) {
    _data_.lsbp_append_covariates( inputs.clustering_covariates_file() );
  }
  if ( !inputs.clustering_random_effects_files().empty() ) {
    for ( std::string fname : inputs.clustering_random_effects_files() ) {
      _data_.lsbp_append_random_effects( fname );
    }
  }
  // _data_.finalize_lsbp_covariates();  // <- testing sparse Z


  // If necessary, estimate covariance parameters by maximizing
  // the marginal log likelihood. Default options only here
  if ( inputs.covariance_parameters().empty() ) {
    std::cout << "Estimating GP covariance parameters:" << std::endl;
    stickygpm::estimate_covariance_parameters( _cov_ptr_, _data_ );
    std::cout << _cov_ptr_->param() << std::endl;
  }

  // std::cout << "X =\n"
  // 	    << _data_.X().topLeftCorner( 10, _data_.X().cols() )
  // 	    << "\n"
  // 	    << "Z =\n"
  // 	    << _data_.Z().topLeftCorner( 10, _data_.Z().cols() )
  // 	    << std::endl;

  // Initialize model object
  stickygpm::stickygpm_regression_model<scalar_type> _model_(
    _data_,
    mask,
    _cov_ptr_,
    inputs.output_file_base(),
    inputs.output_samples(),
    inputs.lsbp_truncation(),
    inputs.lsbp_prior_mu(),
    inputs.lsbp_prior_sigma(),
    inputs.knots(),
    inputs.neighborhood(),
    inputs.repulsion_parameter()
  );

  _model_.run_mcmc(
    _data_,
    inputs.mcmc_burnin(),
    inputs.mcmc_nsave(),
    inputs.mcmc_thin(),
    inputs.output_samples(),
    inputs.monitor()
  );

  _model_.write_posterior_summaries( mask );  
  _model_.close_logs();
  _model_.write_deviance_summary( _data_ );
  
  ::nifti_image_free( mask );
}





