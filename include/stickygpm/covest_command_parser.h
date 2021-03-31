
#include <chrono>
#include <fstream>
#include <iostream>
#include <nlopt.hpp>
#include <stdio.h>
#include <string>
#include <vector>


#ifndef _STICKYGPM_COVEST_COMMAND_PARSER_
#define _STICKYGPM_COVEST_COMMAND_PARSER_


namespace stickygpm {


  template< typename T >
  class covest_command_parser {
  public:
    typedef T scalar_type;
    enum class call_status { success, error, help };
    enum class kernel { radial_basis, matern };

    covest_command_parser( int argc, char **argv );
    bool error() const;
    bool help_invoked() const;
    bool matern() const;
    bool radial_basis() const;
    operator bool() const;
    bool operator!() const;
    double huge_val() const;
    double xtol_rel() const;
    int subsample_size() const;
    nlopt::algorithm algorithm() const;
    std::string algorithm_string() const;
    std::string covariates_file() const;
    std::string kernel_string() const;
    std::string mask_file() const;
    unsigned int seed() const;
    const std::vector<std::string>& data_files() const;

    void show_help() const;
    void show_usage() const;

  private:
    call_status _status;
    kernel _kern;
    double _huge;
    double _xtol;
    int _nsub;
    nlopt::algorithm _alg;
    std::string _covariates_file;
    std::string _mask_file;
    unsigned int _seed;
    std::vector<std::string> _data_files;
  };  

  
};




template< typename T >
void stickygpm::covest_command_parser<T>::show_usage() const {
  std::cerr << "Estimate GP hyperparameters given NIfTI data:\n"
	    << "Usage:\n"
	    << "\tcovest  path/to/data*.nii <options>\n"
	    << std::endl;
};


template< typename T >
void stickygpm::covest_command_parser<T>::show_help() const {
  show_usage();
  std::cerr << "Options:\n"
	    << "  --covariates             file/path  REQUIRED. Mean covars (*.csv) \n"
	    << "  --mask                   file/path  REQUIRED. Analysis mask (*.nii) \n"
	    << "  --bobyqa                            Use BOBYQA optimizer \n"
	    << "  --cobyla                            Use COBYLA optimizer \n"
	    << "  --huge                    float     Large parameter upper bound \n"
	    << "  --matern                            Use Matern kernel \n"
	    << "  --newuoa                            (D) Use NEWUOA optimizer \n"
	    << "  --radial-basis                      (D) Use radial basis kernel \n"
	    << "  --seed                     int      URNG seed \n"
	    << "  --subsample                int      Subsample size (locations) \n"
	    << "  --xtol                    float     Algorithm tolerance \n"
	    << "  -m                       file/path  Alias: --mask \n"
	    << "  -n                         int      Alias: --subsample \n"
	    << "  -X                       file/path  Alias: --covariates \n"
	    << "\n"
	    << "----------------------------------------------------------------------\n"
	    << "Default optimizer is NEWUOA + bound constraints. BOBYQA is usually \n"
	    << "slower, but can give more consistent results. \n"
	    << "\n"
	    << "----------------------------------------------------------------------\n"
	    << std::endl;
};


#include "stickygpm/covest_command_parser.inl"



template< typename T >
stickygpm::covest_command_parser<T>::covest_command_parser(
  int argc,
  char *argv[]
) {
  
  const auto time = std::chrono::high_resolution_clock::now()
    .time_since_epoch();
  std::ifstream ifs;
  
  // --- Default Values ----------------------------------------------
  _status = call_status::success;
  _kern = kernel::radial_basis;
  _alg = nlopt::LN_NEWUOA_BOUND;

  _huge = 100;
  _xtol = 1e-5;
  _nsub = 2048;
  
  _seed = static_cast<unsigned>(
    std::chrono::duration_cast<std::chrono::milliseconds>(time).count()
  );
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
      else if ( arg == "--bobyqa" ) {
	_alg = nlopt::LN_BOBYQA;
      }
      else if ( arg == "--cobyla" ) {
	_alg = nlopt::LN_COBYLA;
      }
      else if ( arg == "--newuoa" ) {
	_alg = nlopt::LN_NEWUOA_BOUND;
      }
      else if ( arg == "--radial-basis" ) {
	_kern = kernel::radial_basis;
      }
      else if ( arg == "--matern" ) {
	_kern = kernel::matern;
      }
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
      else if ( arg == "--huge" ) {
	if ( i + 1 < argc ) {
	  i++;
	  try {
	    _huge = std::stod( argv[i] );
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
      }  // huge
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
      else if ( arg == "--subsample" || arg == "-n" ) {
	if ( i + 1 < argc ) {
	  i++;
	  try {
	    _nsub = std::abs( std::stoi(argv[i]) );
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
      }  // subsample
      else if ( arg == "--xtol" ) {
	if ( i + 1 < argc ) {
	  i++;
	  try {
	    _xtol = std::stod( argv[i] );
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
      }  // xtol
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

  
  if ( help_invoked() ) {
    show_help();
  }
  else {
    if ( _data_files.empty() ) {
      std::cerr << "\n*** ERROR: "
		<< " User must supply input data files (*.nii)\n";
      _status = call_status::error;
    }
    if ( _covariates_file.empty() ) {
      std::cerr << "\n*** ERROR: "
		<< " User must supply input covariate file (as *.csv)\n";
      _status = call_status::error;
    }
    if ( _mask_file.empty() ) {
      std::cerr << "\n*** ERROR: "
		<< " User must supply analysis mask (as *.nii)\n";
      _status = call_status::error;      
    }
  }

  if ( error() ) {
    show_usage();
    std::cerr << "\nSee covest -h or --help for more information\n";
  }
};



#endif  // _STICKYGPM_COVEST_COMMAND_PARSER_
