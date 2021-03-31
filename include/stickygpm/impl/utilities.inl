

#include <algorithm>
// #include <boost/algorithm/string/replace.hpp>
#include <boost/filesystem.hpp>
// #include <boost/regex.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <pwd.h>     // getpwuid
#include <regex>
#include <stdexcept>
#include <stdlib.h>  // getenv
#include <string>
#include <unistd.h>  // getuid
#include <vector>

#include "stickygpm/defines.h"


bool stickygpm::initialize_temporary_directory() {
  std::lock_guard<std::mutex> _lock(stickygpm::__internals::_MTX_);
  boost::filesystem::create_directory(stickygpm::cache_dir());
  return boost::filesystem::is_directory(stickygpm::cache_dir());
  // boost::filesystem::create_directory(stickygpm::__internals::_TEMP_DIR_);
  // return boost::filesystem::is_directory(stickygpm::__internals::_TEMP_DIR_);
};


stickygpm::path stickygpm::fftw_wisdom_file() {
  return stickygpm::__internals::_FFTW_WISDOM_FILE_;
};


stickygpm::__internals::rng_type& stickygpm::rng() {
  return stickygpm::__internals::_RNG_;
};


int stickygpm::set_number_of_threads( const int threads ) {
  if (threads > 0 && threads <= stickygpm::__internals::_MAX_THREADS_) {
    stickygpm::__internals::_N_THREADS_ = threads;
  }
  return stickygpm::__internals::_N_THREADS_;
};


int stickygpm::threads() {
  return stickygpm::__internals::_N_THREADS_;
};
  

void stickygpm::set_seed( const unsigned int seed ) {
  stickygpm::__internals::_RNG_.seed(seed);
};

  



bool stickygpm::utilities::file_exists( const std::string &fname ) {
  std::ifstream ifs(fname.c_str());
  if (ifs.is_open()) {
    ifs.close();
    return true;
  }
  return false;
};



stickygpm::path stickygpm::utilities::home_directory() {
  // #ifdef unix ... or __APPLE__ ...
  std::string home_(getenv("HOME"));
  if (home_.empty()) {
    struct passwd* pw_ = getpwuid(getuid());
    if (!pw_) {
      std::cerr << "HOME environment variable not defined!"
		<< std::endl;
      home_ = "./";
    }
    else {
      home_ = pw_->pw_dir;
    }
  }
  return stickygpm::path(home_);
};



std::vector<std::string> stickygpm::utilities::list_files(
  const std::string path
) {
  const std::string _home_shorthand = "^~([\\\\/]{1,})";
  const std::string _home =
    ( stickygpm::utilities::home_directory().string() +
     stickygpm::path::preferred_separator );
  
  std::string path_st = std::regex_replace(
    path, std::regex(_home_shorthand), _home);
  
  const stickygpm::path _path(path_st);
  stickygpm::path dired_dir = _path.branch_path();
  std::string filt = _path.filename().string();

  std::vector<std::string> matching_files;


  // Replace literal "."s with "\."
  filt = std::regex_replace(
    filt, std::regex("(.*)\\.(.*)"), "$1\\.$2");
  // ... and modify any wildcard characters
  filt = std::regex_replace(filt, std::regex("\\*"), ".*");

  try {

    std::regex file_filter(filt);
    std::smatch ___;
    bool is_file, matches_filter;
    boost::filesystem::directory_iterator dir_end;
      // ^^ default c'tor is past the end

    for ( boost::filesystem::directory_iterator it(dired_dir);
	  it != dir_end; ++it ) {

      is_file = boost::filesystem::is_regular_file( it->status() );
      matches_filter = std::regex_match(
        it->path().filename().string(), ___, file_filter );

      if ( is_file && matches_filter )
	matching_files.push_back( it->path().string() );
      
    } 
  }
  catch (...) {
    std::cout << "Not found: " << dired_dir.string() << std::endl;
    throw std::runtime_error("Could not locate directory");
  }
   
  std::sort(matching_files.begin(), matching_files.end());
  return matching_files;
};




// progress_bar
// -------------------------------------------------------------------


stickygpm::utilities::progress_bar::progress_bar( unsigned int max_val ) {
  _active = true;
  __ = '=';
  
  _max_val = max_val;
  _print_width = 60;
  _bar_print_width = _print_width - 8;  // 8 additional characters: || xy.z%
  _value = 0;
};
      
void stickygpm::utilities::progress_bar::finish() {
  _active = false;
std::cout << std::setprecision(4) << std::endl;
};

void stickygpm::utilities::progress_bar::operator++() {
  _value++;
  _value = (_value > _max_val) ? _max_val : _value;
};

void stickygpm::utilities::progress_bar::operator++(int) {
  ++(*this);
};

void stickygpm::utilities::progress_bar::value(unsigned int value) {
  _value = value;
  _value = (_value > _max_val) ? _max_val : _value;
};
      



template< typename OStream >
OStream& stickygpm::utilities::operator<<(
    OStream& os,
    const stickygpm::utilities::progress_bar& pb
) {
  const double prop = (double)pb._value / pb._max_val;
  const unsigned int bars = (unsigned int)(prop * pb._bar_print_width);
  if (pb._active) {
    if (pb._value > 0) {
      for (unsigned int i = 0; i < pb._print_width; i++)  os << "\b";
    }
    os << "|";
    for (unsigned int i = 1; i <= pb._bar_print_width; i++) {
      if (i <= bars)
	os << pb.__;
      else
	os << " ";
    }
    if ( prop < 0.095 )
      os << "|  ";
    else if ( prop < 0.995 )
      os << "| ";
    else
      os << "|";
    os << std::setprecision(1) << std::fixed << (prop * 100) << "%"
       << std::flush;
  }
  return os;
};



