
#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

#include "stickygpm/covariance_functors.h"
#include "stickygpm/covest_command_parser.h"
#include "stickygpm/estimate_kernel_parameters.h"
#include "stickygpm/utilities.h"
#include "stickygpm/stickygpm_regression_data.h"





int main ( int argc, char* argv[] ) {
#ifdef STICKYGPM_DOUBLE_PRECISION
  typedef double scalar_type;
#else
  typedef float scalar_type;
#endif

  stickygpm::covest_command_parser<scalar_type> inputs(argc, argv);
  if ( !inputs )
    return 1;
  else if ( inputs.help_invoked() )
    return 0;

  stickygpm::set_seed( inputs.seed() );
  std::cout << "[Optimizing "
	    << inputs.kernel_string()
	    << " covariance parameters using "
	    << inputs.algorithm_string()
	    << " algorithm]"
	    << std::endl;
  
  
  std::shared_ptr< stickygpm::covariance_functor<scalar_type> >
    _cov_ptr_;
  if ( inputs.radial_basis() ) {
    _cov_ptr_ =
      std::make_shared< stickygpm::radial_basis<scalar_type> >();
  }
  else if ( inputs.matern() ) {
    _cov_ptr_ =
      std::make_shared< stickygpm::matern<scalar_type> >();
  }
  else {
    std::cerr << "Unrecognized covaraince function type\n";
    return 1;
  }


  stickygpm::stickygpm_regression_data<scalar_type> _data_(
    inputs.data_files(),
    inputs.mask_file(),
    inputs.covariates_file()
  );

  auto start_t = std::chrono::high_resolution_clock::now();
  int code = stickygpm::estimate_covariance_parameters(
    _cov_ptr_,
    _data_,
    inputs.subsample_size(),
    inputs.algorithm(),
    inputs.xtol_rel(),
    inputs.huge_val()
  );
  auto stop_t = std::chrono::high_resolution_clock::now();
  auto diff_t = std::chrono::duration_cast<std::chrono::microseconds>
    ( stop_t - start_t );

  stickygpm::covariance_functor<scalar_type>::param_type
    theta = _cov_ptr_->param();

  
  if ( code == 0 ) {
    std::cout << "<Computation took "
	      << ((double)diff_t.count() / 1e6)
	      << " sec>"
	      << std::endl;
    std::cout << theta << std::endl;
  }
  else {
    std::cerr << "covest: errors occurred" << std::endl;
    return 1;
  }

}  // main





