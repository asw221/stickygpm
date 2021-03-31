
#include <iomanip>
#include <iostream>
#include <nifti1_io.h>
#include <sstream>
#include <stdexcept>
#include <string>

#include "stickygpm/nifti_manipulation.h"


int main (int argc, char* argv[]) {
  typedef float scalar_type;

  if ( argc < 3 ) {
    std::cerr << "\nUsage:\n\tthreshold_nifti /path/to/img T\n";
    return 1;
  }

  bool error_status = false;
  std::ostringstream new_fname_stream;

  const std::string _image_file( argv[1] );
  if ( !stickygpm::is_nifti_file( _image_file ) ) {
    std::cerr << "threshold_nifti: requires NIfTI file as input\n";
    return 1;
  }

  try {
    const scalar_type _threshold = (scalar_type)std::stod( argv[2] );
    ::nifti_image* _nii = stickygpm::nifti_image_read( _image_file, 1 );
    stickygpm::threshold( _nii, _threshold );

    new_fname_stream << std::setprecision(2) << std::fixed
		     << ::nifti_makebasename( _nii->fname ) << "_Thr"
		     << (_threshold) << ".nii";

    stickygpm::nifti_image_write( _nii, new_fname_stream.str() );
    ::nifti_image_free( _nii );
  }
  catch (const std::exception& __err) {
    error_status = true;
    std::cerr << "Exception caught with message:\n'"
	      << __err.what() << "'\n"
	      << std::endl;
  }
  catch (...) {
    error_status = true;
    std::cerr << "Unknown error\n";
  }

  if ( error_status )  return 1;
}




