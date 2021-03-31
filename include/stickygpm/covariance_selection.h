
#include "stickygpm/covariance_functors.h"


#ifndef _STICKYGPM_COVARIANCE_SELECTION_
#define _STICKYGPM_COVARIANCE_SELECTION_

namespace stickygpm {

  enum class cov_options {
    radial_basis,
    matern
  };



  template< typename T, cov_options >
  struct covariance_selection;

  template< typename T >
  struct covariance_selection<cov_options::radial_basis> {
    using type = stickygpm::radial_basis<T>;
  };

  // template< typename T>
  // struct covariance_selection<cov_options::matern> {
  //   using type = stickygpm::matern<T>;
  // };

  template< cov_options CovT, typename T >
  covariance_selection<T, CovT>::type covariance_factory(
    const std::vector<T>& parameters
  ) {
    return covariance_selection<T, CovT>::type(
      parameters.cbegin(),
      parameters.end()
    );
  };
  
  
};


#endif  // _STICKYGPM_COVARIANCE_SELECTION_

