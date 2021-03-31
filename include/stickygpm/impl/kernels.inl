
#include <cmath>
#include <stdexcept>


// scalar version
template< typename T >
T stickygpm::kernels::rbf(
  const T &distance,
  const T &bandwidth,
  const T &exponent,
  const T &variance
) {
  return variance *
    std::exp(-bandwidth * std::pow(std::abs(distance), exponent));
};


template< typename T >
T stickygpm::kernels::rbf_inverse(
  const T &rho,
  const T &bandwidth,
  const T &exponent,
  const T &variance
) {
  if (rho <= (T)0 || rho >= (T)1)
    throw std::domain_error(
      "rbf_inverse: inverse kernel only for rho between (0, 1)");
  return std::pow(-std::log(rho / variance) / bandwidth,
		  1 / exponent);
};




template< typename T >
T stickygpm::kernels::rbf_bandwidth_to_fwhm(
  const T &bandwidth,
  const T &exponent
) {
  return 2.0 * std::pow(std::log((T)2) / bandwidth, 1 / exponent);
};



template< typename T >
T stickygpm::kernels::rbf_fwhm_to_bandwidth(
  const T &fwhm,
  const T &exponent
) {
  return std::log((T)2) / std::pow(fwhm / 2, exponent);
};










