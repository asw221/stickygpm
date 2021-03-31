

#ifndef _STICKYGPM_KERNELS_
#define _STICKYGPM_KERNELS_


namespace stickygpm {
  

  namespace kernels {
    /*! @addtogroup GaussianProcessModels 
     * @{
     */

    // scalar version
    template< typename T >
    T rbf(
      const T &distance,
      const T &bandwidth,
      const T &exponent = 1.9999,
      const T &variance = 1.0
    );


    template< typename T >
    T rbf_inverse(
      const T &rho,
      const T &bandwidth,
      const T &exponent = 1.9999,
      const T &variance = 1.0
    );


    template< typename T >
    T rbf_bandwidth_to_fwhm(const T &bandwidth, const T &exponent = 1.9999);
    
    template< typename T >
    T rbf_fwhm_to_bandwidth(const T &fwhm, const T &exponent = 1.9999);


    /*! @} */
  }  // namespace kernels 
  
}



#include "stickygpm/impl/kernels.inl"

#endif  // _STICKYGPM_KERNELS_


    
