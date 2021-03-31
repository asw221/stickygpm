
#include <string>
#include <vector>


template< typename T >
bool stickygpm::sgpreg_command_parser<T>::error() const {
  return _status == call_status::error;
};


template< typename T >
bool stickygpm::sgpreg_command_parser<T>::help_invoked() const {
  return _status == call_status::help;
};


template< typename T >
bool stickygpm::sgpreg_command_parser<T>::lsbp_intercept() const {
  return _lsbp_intercept;
};


template< typename T >
bool stickygpm::sgpreg_command_parser<T>::monitor() const {
  return _monitor;
};


template< typename T >
bool stickygpm::sgpreg_command_parser<T>::output_samples() const {
  return _output_samples;
};


template< typename T >
stickygpm::sgpreg_command_parser<T>::operator bool() const {
  return !error();
};


template< typename T >
bool stickygpm::sgpreg_command_parser<T>::operator!() const {
  return error();
};


template< typename T >
double stickygpm::sgpreg_command_parser<T>::repulsion_parameter()
  const {
  return _repulsion;
};


template< typename T >
int stickygpm::sgpreg_command_parser<T>::knots() const {
  return _knots;
};


template< typename T >
int stickygpm::sgpreg_command_parser<T>::lsbp_truncation() const {
  return _lsbp_truncation;
};



template< typename T >
int stickygpm::sgpreg_command_parser<T>::mcmc_burnin() const {
  return _mcmc_burnin;
};


template< typename T >
int stickygpm::sgpreg_command_parser<T>::mcmc_nsave() const {
  return _mcmc_nsave;
};


template< typename T >
int stickygpm::sgpreg_command_parser<T>::mcmc_thin() const {
  return _mcmc_thin;
};


template< typename T >
int stickygpm::sgpreg_command_parser<T>::threads() const {
  return _threads;
};


template< typename T >
int stickygpm::sgpreg_command_parser<T>::neighborhood() const {
  return _neighborhood;
};



template< typename T >
typename stickygpm::sgpreg_command_parser<T>::scalar_type
stickygpm::sgpreg_command_parser<T>::lsbp_prior_mu() const {
  return _lsbp_mu;
};


template< typename T >
typename stickygpm::sgpreg_command_parser<T>::scalar_type
stickygpm::sgpreg_command_parser<T>::lsbp_prior_sigma() const {
  return _lsbp_sigma;
};




template< typename T >
std::string
stickygpm::sgpreg_command_parser<T>::clustering_covariates_file() const {
  return _clustering_covariates_file;
};


template< typename T >
std::string
stickygpm::sgpreg_command_parser<T>::covariates_file() const {
  return _covariates_file;
};


template< typename T >
const std::vector<std::string>&
stickygpm::sgpreg_command_parser<T>::data_files() const {
  return _data_files;
};


template< typename T >
std::string
stickygpm::sgpreg_command_parser<T>::mask_file() const {
  return _mask_file;
};


template< typename T >
std::string
stickygpm::sgpreg_command_parser<T>::output_file_base() const {
  return _output_basename;
};


template< typename T >
std::string stickygpm::sgpreg_command_parser<T>::output_file(
  const std::string extension
) const {
  return _output_basename + extension;
};


template< typename T >
unsigned int stickygpm::sgpreg_command_parser<T>::seed() const {
  return _seed;
};


template< typename T >
const std::vector<T>&
stickygpm::sgpreg_command_parser<T>::covariance_parameters() const {
  return _theta;
};


template< typename T >
const std::vector<std::string>&
stickygpm::sgpreg_command_parser<T>
::clustering_random_effects_files() const {
  return _clustering_random_effects_files;
};


