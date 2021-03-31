
#include <algorithm>
#include <boost/filesystem.hpp>
#include <mutex>
#include <random>
#include <stdlib.h>  // getenv
#include <string>
#include <thread>


#ifndef _STICKYGPM_DEFINES_
#define _STICKYGPM_DEFINES_


namespace stickygpm {
  /*! @addtogroup Stickygpm
   * @{
   */
  

  typedef boost::filesystem::path path;


  /*!
   * Unused, currently.
   */
  enum class use_lambda_method { PROFILE, NATIVE, EXTENDED };


  /*!
   * See official NIfTI datatype 
   * <a href="https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/datatype.html">definitions</a>
   */
  enum class nifti_data_type {
    OTHER = 0,  BINARY = 1,   CHAR = 2,    SHORT = 4, INT = 8,
    FLOAT = 16, COMPLEX = 32, DOUBLE = 64, RGB = 128, ALL = 255
  };



  stickygpm::path cache_dir() {
    const char* user_home_cstr = getenv("HOME");
    std::string _cache_d;
    if (user_home_cstr == NULL) {
      _cache_d = boost::filesystem::temp_directory_path().string() +
	stickygpm::path::preferred_separator +
	"stickygpmTemp";
    }
    else {
      _cache_d = std::string(user_home_cstr) +
	stickygpm::path::preferred_separator +
	".stickygpm.cache";
    }
    return stickygpm::path(_cache_d);
  };


  

  /// @cond INTERNAL
  namespace __internals {

    typedef std::mt19937 rng_type;
    // typedef boost::filesystem::path path;


    bool _MONITOR_ = false;

    const int _MAX_THREADS_ = std::thread::hardware_concurrency();
    int _N_THREADS_ = std::max(_MAX_THREADS_ * 4 / 5, 1);

    std::mutex _MTX_;
    
    rng_type _RNG_(42);

    

    const stickygpm::path _TEMP_DIR_(
      boost::filesystem::temp_directory_path().string() +
      stickygpm::path::preferred_separator + "stickygpmTemp" +
      stickygpm::path::preferred_separator
    );

    
#ifdef STICKYGPM_SINGLE_PRECISION
    const stickygpm::path _FFTW_WISDOM_FILE_(
      stickygpm::cache_dir().string() +
      stickygpm::path::preferred_separator +
      "__fftwf_wisdom"
    );
    // _TEMP_DIR_.string() + "__fftwf_wisdom"
#else
    const stickygpm::path _FFTW_WISDOM_FILE_(
      stickygpm::cache_dir().string() +
      stickygpm::path::preferred_separator +
      "__fftw_wisdom"
    );
    // _TEMP_DIR_.string() + "__fftw_wisdom"
#endif
    
  }  // namespace __internals
  /// @endcond

  

  stickygpm::path current_path() {
    return boost::filesystem::current_path();
  };

  void current_path(const stickygpm::path &p) {
    if (!p.empty())
      boost::filesystem::current_path(p);
  };

  /*! @} */  
}
// namespace stickygpm


#endif  // _STICKYGPM_DEFINES_





