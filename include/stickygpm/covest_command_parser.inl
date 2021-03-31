
#include <nlopt.hpp>
#include <string>
#include <vector>




template< typename T >
bool stickygpm::covest_command_parser<T>::error() const {
  return _status == call_status::error;
};


template< typename T >
bool stickygpm::covest_command_parser<T>::help_invoked() const {
  return _status == call_status::help;
};


template< typename T >
bool stickygpm::covest_command_parser<T>::matern() const {
  return _kern == kernel::matern;
};


template< typename T >
bool stickygpm::covest_command_parser<T>::radial_basis() const {
  return _kern == kernel::radial_basis;
};


template< typename T >
stickygpm::covest_command_parser<T>::operator bool() const {
  return !error();
};


template< typename T >
bool stickygpm::covest_command_parser<T>::operator!() const {
  return error();
};



template< typename T >
double stickygpm::covest_command_parser<T>::huge_val() const {
  return _huge;
};


template< typename T >
double stickygpm::covest_command_parser<T>::xtol_rel() const {
  return _xtol;
};


template< typename T >
int stickygpm::covest_command_parser<T>::subsample_size() const {
  return _nsub;
};


template< typename T >
nlopt::algorithm
stickygpm::covest_command_parser<T>::algorithm() const {
  return _alg;
};


template< typename T >
std::string
stickygpm::covest_command_parser<T>::covariates_file() const {
  return _covariates_file;
};




template< typename T >
std::string
stickygpm::covest_command_parser<T>::algorithm_string() const {
  if ( _alg == nlopt::LN_COBYLA ) {
    return std::string( "COBYLA" );
  }
  else if ( _alg == nlopt::LN_BOBYQA ) {
    return std::string( "BOBYQA" );
  }
  else if ( _alg == nlopt::LN_NEWUOA_BOUND ) {
    return std::string( "(Bounded) NEWUOA" );
  }
  
  return std::string( "Unknown" );
};



template< typename T >
std::string
stickygpm::covest_command_parser<T>::kernel_string() const {
  if ( _kern == kernel::radial_basis ) {
    return std::string( "Radial basis" );
  }
  else if ( _kern == kernel::matern ) {
    return std::string( "Matern" );
  }

  return std::string( "Unknown" );
};




template< typename T >
std::string
stickygpm::covest_command_parser<T>::mask_file() const {
  return _mask_file;
};


template< typename T >
unsigned int stickygpm::covest_command_parser<T>::seed() const {
  return _seed;
};



template< typename T >
const std::vector< std::string >&
stickygpm::covest_command_parser<T>::data_files() const {
  return _data_files;
};



