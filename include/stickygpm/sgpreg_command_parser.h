
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>

#include "stickygpm/constants.h"

#ifndef _STICKYGPM_SGPREG_COMMAND_PARSER_
#define _STICKYGPM_SGPREG_COMMAND_PARSER_


namespace stickygpm {


  template< typename T >
  class sgpreg_command_parser {
  public:
    typedef T scalar_type;
    enum class call_status { success, error, help };

    sgpreg_command_parser( int argc, char **argv );
    bool error() const;
    bool help_invoked() const;
    bool lsbp_intercept() const;
    bool monitor() const;
    bool output_samples() const;
    bool radial_basis() const { return true; };
    operator bool() const;
    bool operator!() const;
    int knots() const;
    // double rejection_rate() const;
    double repulsion_parameter() const;
    int initial_k() const;
    int lsbp_truncation() const;
    int mcmc_burnin() const;
    int mcmc_nsave() const;
    int mcmc_thin() const;
    int threads() const;
    int neighborhood() const;
    scalar_type lsbp_prior_mu() const;
    scalar_type lsbp_prior_sigma() const;
    std::string clustering_covariates_file() const;
    std::string covariates_file() const;
    // std::string data_file_pattern() const;
    std::string mask_file() const;
    std::string output_file_base() const;
    std::string output_file( const std::string extension ) const;
    std::string subset_file() const;
    unsigned int seed() const;
    const std::vector<T>& covariance_parameters() const;
    const std::vector<std::string>& clustering_random_effects_files() const;
    const std::vector<std::string>& data_files() const;

    void show_help() const;
    void show_usage() const;

  private:
    call_status _status;
    bool _lsbp_intercept;
    bool _monitor;
    bool _output_samples;
    // double _rejection_rate;
    double _repulsion;
    int _init_k;
    int _knots;
    int _lsbp_truncation;
    int _mcmc_burnin;
    int _mcmc_nsave;
    int _mcmc_thin;
    int _threads;
    int _neighborhood;
    scalar_type _lsbp_mu;
    scalar_type _lsbp_sigma;
    std::string _clustering_covariates_file;
    std::string _covariates_file;
    std::string _subset_file;
    // std::string _data_file_pattern;
    std::string _mask_file;
    std::string _output_basename;
    unsigned int _seed;
    std::vector<T> _theta;
    std::vector<std::string> _clustering_random_effects_files;
    std::vector<std::string> _data_files;

    void _subset_outcome_files();
  };  

  
};




template< typename T >
void stickygpm::sgpreg_command_parser<T>::show_usage() const {
  std::cerr << "Fit clustered spatial regression models to NIfTI data:\n"
	    << "Usage:\n"
	    << "\tsgpreg  path/to/data*.nii <options>\n"
	    << std::endl;
};


template< typename T >
void stickygpm::sgpreg_command_parser<T>::show_help() const {
  show_usage();
  std::cerr << "Options:\n"
	    << "  --covariates             file/path  REQUIRED. Mean covars (*.csv) \n"
	    << "  --mask                   file/path  REQUIRED. Analysis mask (*.nii) \n"
	    << "  --burnin                   int      MCMC burnin iterations \n"
	    << "  --clustering-covariates  file/path  LSBP fixed effects (*.csv) \n"
	    << "  --covariance             float...   Spatial cov. func. parameters \n"
	    << "  --debug                             Flag: short MCMC chain + output \n"
	    << "  --kmeans                   int      Initial clustering K \n"
	    << "  --knots                    int      Number of GP bases \n"
	    << "  --lsbp-mu                 float     LSBP intercept hyper \n"
	    << "  --lsbp-no-intercept                 LSBP don't include intercept \n"
	    << "  --lsbp-sigma              float     LSBP coefficients hyper \n"
	    << "  --monitor                           Flag: verbose messaging \n"
	    << "  --neighborhood             int      Sparse cov. func. n'hood size \n"
	    << "  --nsave                    int      MCMC samples to save \n"
	    << "  --output                 basename   Basename for output files \n"
	    << "  --random                 file/path  LSBP random effects (*.csv) \n"
	    << "  --random-effects         file/path  Alias: --random \n"
	    << "  --repulsion               float     Repulsive mixture parameter \n"
	    << "  --samples                           MCMC/Flag: output 'all' samples \n"
	    << "  --seed                     int      URNG seed \n"
	    << "  --subset                 file/path  File: tokens to select outcomes \n"
	    << "  --theta                   float...  Alias: --covariance \n"
	    << "  --thin                     int      MCMC thinning factor \n"
	    << "  --threads                  int      Threads to use (OpenMP) \n"
	    << "  --truncate                 int      LSBP max clusters \n"
	    << "  -k                         int      Alias: --neighborhood \n"
	    << "  -m                       file/path  Alias: --mask \n"
	    << "  -o                       basename   Alias: --output \n"
	    << "  -re                      file/path  Alias: --random \n"
	    << "  -T                         int      Alias: --truncate \n"
	    << "  -X                       file/path  Alias: --covariates \n"
	    << "  -Z                       file/path  Alias: --clustering-covariates \n"
	    << "\n"
	    << "----------------------------------------------------------------------\n"
	    << "----------------------------------------------------------------------\n"
	    << std::endl;
};


#include "stickygpm/sgpreg_command_parser.inl"



template< typename T >
stickygpm::sgpreg_command_parser<T>::sgpreg_command_parser(
  int argc,
  char *argv[]
) {
  
  const auto time = std::chrono::high_resolution_clock::now()
    .time_since_epoch();
  std::ifstream ifs;
  std::ofstream ofs;
  std::stringstream sstream;
  bool seek;
  
  // --- Default Values ----------------------------------------------
  _status = call_status::success;

  _init_k = -1;
  _knots = 2000;
  _lsbp_truncation = 10;
  _lsbp_mu = 0;
  // _lsbp_mu = -Q_LOGIS_0_05;
  _lsbp_intercept = true;
  _lsbp_sigma = SQRT_LN2_2;
  // _rejection_rate = 0.5;
  _repulsion = 0;
  
  _mcmc_burnin = 500;
  _mcmc_nsave = 1000;
  _mcmc_thin = 5;
  _monitor = false;
  
  _neighborhood = 2000;
  _output_samples = false;
  _seed = static_cast<unsigned>(
    std::chrono::duration_cast<std::chrono::milliseconds>(time).count()
  );
  _threads = static_cast<unsigned>(0);
  
  // -----------------------------------------------------------------


  if ( argc < 2 ) {
    _status = call_status::error;
  }
  else {
    for ( int i = 1; i < argc; i++) {
      std::string arg = argv[ i ];
      if ( arg == "-h" || arg == "--help" ) {
	_status = call_status::help;
	break;
      }
      else if ( arg == "--burnin" ) {
	if ( (i + 1) < argc ) {
	  i++;
	  try {
	    _mcmc_burnin = std::abs( std::stoi(argv[i]) );
	    _mcmc_burnin = ( _mcmc_burnin < 1) ? 1 : _mcmc_burnin;
	  }
	  catch (...) {
	    std::cerr << arg << " option requires one integer argument\n";
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << arg << " option requires one integer argument\n";
	  _status = call_status::error;
	}
      }  // burnin
      else if ( arg == "--clustering-covariates" || arg == "-Z" ) {
	if ( i + 1 < argc ) {
	  i++;
	  _clustering_covariates_file = argv[i];
	  ifs.open( _clustering_covariates_file, std::ifstream::in );
	  if ( !ifs ) {
	    std::cerr << "\nCould not read "
		      << _clustering_covariates_file << "\n";
	    ifs.clear();
	    _status = call_status::error;
	  }
	  ifs.close();
	}
	else {
	  std::cerr << arg << " option requires one argument\n";
	  _status = call_status::error;
	}
      }  // clustering covariates
      else if ( arg == "--covariates" || arg == "-X" ) {
	if ( i + 1 < argc ) {
	  i++;
	  _covariates_file = argv[i];
	  ifs.open( _covariates_file, std::ifstream::in );
	  if ( !ifs ) {
	    std::cerr << "\nCould not read "
		      << _covariates_file << "\n";
	    ifs.clear();
	    _status = call_status::error;
	  }
	  ifs.close();
	}
	else {
	  std::cerr << arg << " option requires one argument\n";
	  _status = call_status::error;
	}
      }  // covariates
      else if ( arg == "--covariance" || arg == "--theta" ) {
	seek = true;
	while ( (i + 1) < argc && seek ) {
	  i++;
	  try {
	    _theta.push_back(
              static_cast<scalar_type>(std::stod( argv[i] ))
            );
	  }
	  catch (...) {
	    seek = false;
	    i--;
	  }
	}
      }  // covariance
      else if ( arg == "--debug" ) {
	_mcmc_burnin = 100;
	_mcmc_nsave = 100;
	_mcmc_thin = 1;
	_monitor = true;
	_output_samples = true;
      }  // debug
      else if ( arg == "--kmeans" ) {
	if ( i + 1 < argc ) {
	  i++;
	  try {
	    _init_k = std::abs( std::stoi(argv[i]) );
	  }
	  catch (...) {
	    std::cerr << arg
		      << " option requires one integer argument\n";
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << arg
		    << " option requires one integer argument\n";
	  _status = call_status::error;
	}
      }  // kmeans
      else if ( arg == "--knots" ) {
	if ( i + 1 < argc ) {
	  i++;
	  try {
	    _knots = std::abs( std::stoi(argv[i]) );
	  }
	  catch (...) {
	    std::cerr << arg
		      << " option requires one integer argument\n";
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << arg
		    << " option requires one integer argument\n";
	  _status = call_status::error;
	}
      }  // knots
      else if ( arg == "--lsbp-mu" ) {
	if ( i + 1 < argc ) {
	  i++;
	  try {
	    _lsbp_mu = static_cast<scalar_type>( std::stod(argv[i]) );
	  }
	  catch (...) {
	    std::cerr << arg
		      << " option requires one floating point argument\n";
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << arg
		    << " option requires one floating point argument\n";
	  _status = call_status::error;
	}
      }  // lsbp-mu
      else if ( arg == "--lsbp-no-intercept" ) {
	_lsbp_intercept = false;
      }  // lsbp-no-intercept
      else if ( arg == "--lsbp-sigma" ) {
	if ( i + 1 < argc ) {
	  i++;
	  try {
	    _lsbp_sigma = std::abs(
              static_cast<scalar_type>(std::stod( argv[i] ))
            );
	  }
	  catch (...) {
	    std::cerr << arg
		      << " option requires one floating point argument\n";
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << arg
		    << " option requires one floating point argument\n";
	  _status = call_status::error;
	}
      }  // lsbp-sigma
      else if ( arg == "--mask" || arg == "-m" ) {
	if ( i + 1 < argc ) {
	  i++;
	  _mask_file = argv[i];
	  ifs.open( _mask_file, std::ifstream::in );
	  if ( !ifs ) {
	    std::cerr << "\nCould not read " << _mask_file << "\n";
	    ifs.clear();
	    _status = call_status::error;
	  }
	  ifs.close();
	}
	else {
	  std::cerr << arg << " option requires one argument\n";
	  _status = call_status::error;
	}
      }  // mask
      else if ( arg == "--monitor" || arg == "--verbose" ) {
	_monitor = true;
      }
      else if ( arg == "--neighborhood" || arg == "-k" ) {
	if ( i + 1 < argc ) {
	  i++;
	  try {
	    _neighborhood = std::abs( std::stoi(argv[i]) );
	  }
	  catch (...) {
	    std::cerr << arg
		      << " option requires one integer argument\n";
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << arg
		    << " option requires one integer argument\n";
	  _status = call_status::error;
	}
      }  // neighborhood
      else if ( arg == "--nsave" ) {
	if ( i + 1 < argc ) {
	  i++;
	  try {
	    _mcmc_nsave = std::abs( std::stoi(argv[i]) );
	    _mcmc_nsave = ( _mcmc_nsave < 1 ) ? 1 : _mcmc_nsave;
	  }
	  catch (...) {
	    std::cerr << arg
		      << " option requires one integer argument\n";
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << arg
		    << " option requires one integer argument\n";
	  _status = call_status::error;
	}
      }  // nsave
      else if ( arg == "--output" || arg == "-o" ) {
	if ( i + 1 < argc ) {
	  i++;
	  _output_basename = argv[i];
	  ofs.open( _output_basename, std::ifstream::out );
	  if ( !ofs ) {
	    std::cerr << "\nOutput base not writable: "
		      << _output_basename << "\n";
	    _status = call_status::error;
	  }
	  else {
	    ofs.close();
	    remove( _output_basename.c_str() );
	  }
	}
	else {
	  std::cerr << arg
		    << " option requires one argument\n";
	  _status = call_status::error;
	}
      }  // output basename
      else if ( arg == "--random-effects" ||
		arg == "--random-effect" ||
		arg == "--random" ||
		arg == "-re"
		) {
	seek = true;
	while ( (i + 1) < argc && seek ) {
	  i++;
	  std::string file = argv[ i ];
	  ifs.open( file );
	  if ( ifs ) {
	    ifs.close();
	    _clustering_random_effects_files.push_back( file );
	  }
	  else {
	    ifs.clear();
	    seek = false;
	    i--;
	  }
	}
	if ( _clustering_random_effects_files.empty() ) {
	  std::cerr << arg << " option requires at least"
		    << " one input argument\n";
	  _status = call_status::error;
	}
      }  // clustering random effects
      else if ( arg == "--repulsion" ) {
	if ( i + 1 < argc ) {
	  i++;
	  try {
	    _repulsion = std::stod( argv[i] );
	    if ( _repulsion < 0 ) {
	      _status = call_status::error;
	      std::cerr << arg
			<< " argument should be > 0\n";
	    }
	  }
	  catch (...) {
	    std::cerr << arg
		      << " option requires one floating point argument\n";
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << arg
		    << " option requires one floating point argument\n";
	  _status = call_status::error;
	}
      }  // rejection-rate
      else if ( arg == "--samples" ) {
	_output_samples = true;
      }
      else if ( arg == "--seed" ) {
	if ( i + 1 < argc ) {
	  i++;
	  try {
	    _seed = static_cast<unsigned>(
              std::abs(std::stoi( argv[i] ))
            );
	  }
	  catch (...) {
	    std::cerr << arg
		      << " option requires one integer argument\n";
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << arg
		    << " option requires one integer argument\n";
	  _status = call_status::error;
	}
      }  // seed
      else if ( arg == "--subset" ) {
	if ( i + 1 < argc ) {
	  i++;
	  _subset_file = argv[i];
	  ifs.open( _subset_file, std::ifstream::in );
	  if ( !ifs ) {
	    std::cerr << "\nCould not read " << _subset_file << "\n";
	    ifs.clear();
	    _status = call_status::error;
	  }
	  ifs.close();
	}
	else {
	  std::cerr << arg << " option requires one argument\n";
	  _status = call_status::error;
	}
      }  // subset
      else if ( arg == "--thin" ) {
	if ( i + 1 < argc ) {
	  i++;
	  try {
	    _mcmc_thin = std::abs(std::stoi( argv[i] ));
	    _mcmc_thin = ( _mcmc_thin < 1 ) ? 1 : _mcmc_thin;
	  }
	  catch (...) {
	    std::cerr << arg
		      << " option requires one integer argument\n";
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << arg
		    << " option requires one integer argument\n";
	  _status = call_status::error;
	}
      }  // thin
      else if ( arg == "--threads" ) {
	if ( i + 1 < argc ) {
	  i++;
	  try {
	    _threads = std::abs(std::stoi( argv[i] ));
	  }
	  catch (...) {
	    std::cerr << arg
		      << " option requires one integer argument\n";
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << arg
		    << " option requires one integer argument\n";
	  _status = call_status::error;
	}
      }  // threads
      else if ( arg == "--truncate" || arg == "-T" ) {
	if ( i + 1 < argc ) {
	  i++;
	  try {
	    _lsbp_truncation = std::abs( std::stoi(argv[i]) );
	    _lsbp_truncation = (_lsbp_truncation == 0) ? 1 :
	      _lsbp_truncation;
	  }
	  catch (...) {
	    std::cerr << arg
		      << " option requires one integer argument\n";
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << arg
		    << " option requires one integer argument\n";
	  _status = call_status::error;
	}
      }  // truncate
      else if ( arg.substr(0, 1) == "-" ) {
	std::cerr << "Unrecognized option '" << arg << "'\n";
      }
      else {
	_data_files.push_back( arg );
      }
      
      // --- end parsing options

      if ( error() || help_invoked() ) {
	break;
      }
    }
    // end - for ( int i = 1; i < argc; i++)
  }
  // end - processing input arguments

  
  if ( _output_basename.empty() ) {
    sstream << "sgpreg_" << _seed;
    _output_basename = sstream.str();
  }

  _subset_outcome_files();
  
  if ( help_invoked() ) {
    show_help();
  }
  else {
    if ( _data_files.empty() ) {
      if ( _subset_file.empty() ) {
	std::cerr << "\n*** ERROR: "
		  << " User must supply input data files (*.nii)\n\n";
      }
      else {
	std::cerr << "\n*** ERROR: "
		  << " No files match --subset tokens\n";
      }
      _status = call_status::error;
    }
    if ( _covariates_file.empty() ) {
      std::cerr << "\n*** ERROR: "
		<< " User must supply input covariate file (as *.csv)\n\n";
      _status = call_status::error;
    }
    if ( _mask_file.empty() ) {
      std::cerr << "\n*** ERROR: "
		<< " User must supply analysis mask (as *.nii)\n\n";
      _status = call_status::error;      
    }
  }


  if ( error() ) {
    show_usage();
    std::cerr << "See sgpreg -h or --help for more information\n";
  }
};



#endif  // _STICKYGPM_SGPREG_COMMAND_PARSER_
