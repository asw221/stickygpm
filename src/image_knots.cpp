
#include <Eigen/Core>
#include <iostream>
#include <nifti1_io.h>
#include <sstream>
#include <stdexcept>
#include <string>

#include "stickygpm/knots.h"
#include "stickygpm/nifti_manipulation.h"
#include "stickygpm/utilities.h"




int main (int argc, char* argv[]) {  
  if ( argc < 2 ) {
    std::cerr << "\nUsage:\n\timage_knots /path/to/mask.nii <nknots> <seed>\n";
    return 1;
  }

  const std::string _image_file( argv[1] );
  int nknots = 2048;
  bool error_status = false;
  std::ostringstream new_fname_stream;
  
  if ( !stickygpm::is_nifti_file( _image_file ) ) {
    std::cerr << "image_knots: requires NIfTI file as input\n";
    return 1;
  }



  try {
    if ( argc >= 3 ) {
      nknots = std::stoi( argv[2] );
    }
    if ( argc >= 4 ) {
      stickygpm::set_seed( (unsigned)std::stoi( argv[3] ) );
    }
    
    ::nifti_image* _mask =
      stickygpm::nifti_image_read(_image_file, 1);
    const bool _mask_uses_double = stickygpm::is_double(_mask);
    if ( !(_mask_uses_double || stickygpm::is_float(_mask)) ) {
      std::cerr << "image_knots: unrecognized image data type\n";
      ::nifti_image_free(_mask);
      return 1;
    }
    
    Eigen::MatrixXf Knots =
      stickygpm::get_knot_positions_uniform<float>(_mask, nknots);
    const float _knot_mmxd =
      stickygpm::minimax_knot_distance(_mask, Knots);
    Eigen::VectorXd _new_data =
      Eigen::VectorXd::Zero((int)_mask->nvox);
    
    new_fname_stream << ::nifti_makebasename(_mask->fname)
		     << "_" << (Knots.rows())
		     << "knots.nii";

    Knots.conservativeResize(Knots.rows(), Knots.cols() + 1);
    Knots.row(Knots.rows() - 1) = Eigen::VectorXf::Ones(Knots.rows());
    Eigen::MatrixXi ijk =
      (Knots * stickygpm::qform_matrix(_mask).inverse().transpose())
      .template cast<int>();
    Knots.conservativeResize(Knots.rows(), Knots.cols() - 1);
    int stride;
    for (int i = 0; i < ijk.rows(); i++) {
      stride = _mask->nx * _mask->ny * ijk.coeffRef(i, 2) +
	_mask->nx * ijk.coeffRef(i, 1) + ijk.coeffRef(i, 0);
      _new_data.coeffRef(stride) = 1;
    }

    // Overwrite data:
    for (int i = 0; i < _new_data.size(); i++) {
      if (_mask_uses_double) {
	*((double*)_mask->data + i) = _new_data.coeffRef(i);
      }
      else {
	*((float*)_mask->data + i) = (float)_new_data.coeffRef(i);
      }
    }

    
    stickygpm::nifti_image_write(_mask, new_fname_stream.str());
    std::cout << "Wrote: " << new_fname_stream.str() << "\n"
	      << "Minimax knot distance: " << _knot_mmxd << " (mm)"
	      << std::endl;
    
    ::nifti_image_free(_mask);
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

  
  if (error_status)  return 1;
}

