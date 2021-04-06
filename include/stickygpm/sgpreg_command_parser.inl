
#include <iostream>
#include <string>
#include <vector>

#include "stickygpm/csv_reader.h"



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
std::string stickygpm::sgpreg_command_parser<T>::subset_file() const {
  return _subset_file;
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



template< typename T >
void stickygpm::sgpreg_command_parser<T>::_subset_outcome_files() {
  if ( !_subset_file.empty() && !_data_files.empty() ) {

    std::vector< std::vector<std::string> > tokens =
      stickygpm::csv_reader<std::string>::read_file( _subset_file );
    
    std::vector<std::string> subset;
    subset.reserve( _data_files.size() );
    // ^^ Reserve to size of _data_files for cases when multiple
    // files match one token, etc

    int matched_tokens = 0, nmatches;
    
    for ( int i = 0; i < (int)tokens.size(); i++ ) {
      // Order the new _data_files in token order
      nmatches = 0;
      
      for ( int j = 0; j < (int)_data_files.size(); j++ ) {
	
	if ( _data_files[j].find( tokens[i][0] ) != std::string::npos ) {
	  if ( subset.empty() ) {
	    subset.push_back( _data_files[j] );
	    nmatches++;
	  }
	  else {
	    // Only push_back _data_files[j] if not already in subset
	    int k = 0;
	    bool duplicated = false;
	    while ( k < (int)subset.size() && !duplicated ) {
	      duplicated = subset[k] == _data_files[j];
	      k++;
	    }
	    if ( !duplicated ) {
	      subset.push_back( _data_files[j] );
	      nmatches++;
	    }
	  }
	}
	// end - if ( _data_files[j].find( ...
	
      }  // end - for ( int j = 0; ...

      if ( nmatches > 0 ) {
	matched_tokens++;
      }
    }    // end - for ( int i = 0; ...

    subset.shrink_to_fit();
    _data_files.assign( subset.begin(), subset.end() );
    if ( _data_files.empty() ) {
      _status = call_status::error;
    }

    if ( matched_tokens < (int)tokens.size() ) {
      std::cerr << "\n\t*** "
		<< matched_tokens << " out of "
		<< tokens.size() << " tokens matched from file "
		<< _subset_file
		<< "\n";
    }
    
  }
  // end - if ( !_subset_file.empty() && ...
};
