
#include <fstream>
#include <string>
#include <vector>


#include "stickygpm/defines.h"


#ifndef _STICKYGPM_UTILITIES_
#define _STICKYGPM_UTILITIES_




namespace stickygpm {
  /*! @addtogroup Stickygpm
   * @{
   */

  bool initialize_temporary_directory();
  
  stickygpm::path fftw_wisdom_file();
  stickygpm::__internals::rng_type& rng();
  
  int set_number_of_threads( const int threads );
  int threads();
  
  void set_seed( const unsigned int seed );

  void set_monitor_simulations( const bool monitor ) {
    stickygpm::__internals::_MONITOR_ = monitor;
  };

  bool monitor_simulations() {
    return stickygpm::__internals::_MONITOR_;
  };


  

  /*! @} */

  

  namespace utilities {
    /*! @addtogroup Stickygpm
     * @{
     */

    
    bool file_exists( const std::string &fname );
    
    stickygpm::path home_directory();


    std::vector<std::string> list_files( const std::string dir );


    


    class progress_bar {
    public:
      progress_bar( unsigned int max_val );
      
      void finish();
      void operator++();
      void operator++( int );
      void value( unsigned int value );

      template< typename OStream >
      friend OStream& operator<<(
        OStream& os,
	const progress_bar& pb
      );
      
    private:
      bool _active;
      char __;
      unsigned int _max_val;
      unsigned int _print_width;
      unsigned int _bar_print_width;
      unsigned int _value;
    };

    
    /*! @} */    
  }  // namespace utilities ------------------------------------------



}  // namespace stickygpm ----------------------------------------------


#include "stickygpm/impl/utilities.inl"

#endif  // _STICKYGPM_UTILITIES_
