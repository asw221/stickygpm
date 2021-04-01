
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <iostream>
#include <nifti1_io.h>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

#include "stickygpm/eigen_read_csv.h"
#include "stickygpm/nifti_manipulation.h"
#include "stickygpm/utilities.h"


#ifndef _STICKYGPM_REGRESSION_DATA_
#define _STICKYGPM_REGRESSION_DATA_


namespace stickygpm {
  

  template< typename RealType >
  class stickygpm_regression_data {
  public:
    typedef std::vector<Eigen::VectorXi> lsbp_index_type;
    typedef RealType scalar_type;
    typedef typename
      Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>
      matrix_type;
    // typedef typename
    //   Eigen::SparseMatrix<scalar_type, Eigen::RowMajor>
    //   sparse_matrix_type;
    typedef typename
      Eigen::Matrix<scalar_type, Eigen::Dynamic, 1>
      vector_type;


    stickygpm_regression_data(
      const std::vector<std::string>& outcome_files,
      const std::string mask_file,
      const std::string covariates_file,
      const bool lsbp_intercept = true
    );

    int n() const;

    const matrix_type& outcome_data() const;
    const matrix_type& Y() const;
    const matrix_type& regression_covariates() const;
    const matrix_type& X() const;
    
    const matrix_type& lsbp_covariates() const;
    const matrix_type& Z() const;

    const matrix_type& mask_locations() const;

    const lsbp_index_type& lsbp_random_effects_indices() const;
    
    bool lsbp_has_global_intercept() const;
    

    void lsbp_append_covariates(
      const std::string covariates_file
    );
    void lsbp_append_random_effects(
      const std::string covariates_file
    );
    void finalize_lsbp_covariates( const RealType eps = 1e-4 );
    
    
  private:
    bool _Z_has_intercept;
    
    matrix_type _Y;  // Extracted image data (subjects across columns)
    matrix_type _X;  // Primary regression covariates 
    matrix_type _Z;  // LSBP inducing covariates
    // matrix_type _Zbuff;     // Buffer to help construct _Z (below)
    

    // std::string _image_data_files_pattern;
    std::string _image_mask_file;

    matrix_type _image_mask_locations;

    std::vector<Eigen::VectorXi> _Z_random_effects_indices;

    void _read_outcome_data(
      const std::vector<std::string>& outcome_files,
      const std::string mask_file
    );
  };


};







template< typename RealType >
stickygpm::stickygpm_regression_data<RealType>::stickygpm_regression_data(
  const std::vector<std::string>& outcome_files,
  const std::string mask_file,
  const std::string covariates_file,
  const bool lsbp_intercept
) {
  // _image_data_files_pattern = outcome_files_pattern;
  _image_mask_file = mask_file;
  _Z_has_intercept = lsbp_intercept;

  _X = stickygpm::read_csv<scalar_type>( covariates_file );
  std::cout << "Read: " << covariates_file << std::endl;
  if ( lsbp_intercept ) {
    _Z = matrix_type::Constant(n(), 1, 1);
  }

  _read_outcome_data( outcome_files, mask_file );
};




template< typename RealType >
int stickygpm::stickygpm_regression_data<RealType>::n() const {
  return _X.rows();
};



template< typename RealType >
const typename stickygpm::stickygpm_regression_data<RealType>
::matrix_type&
stickygpm::stickygpm_regression_data<RealType>
::outcome_data() const {
  return _Y;
};


template< typename RealType >
const typename stickygpm::stickygpm_regression_data<RealType>
::matrix_type&
stickygpm::stickygpm_regression_data<RealType>
::Y() const {
  return _Y;
};




template< typename RealType >
const typename stickygpm::stickygpm_regression_data<RealType>
::matrix_type&
stickygpm::stickygpm_regression_data<RealType>
::regression_covariates() const {
  return _X;
};


template< typename RealType >
const typename stickygpm::stickygpm_regression_data<RealType>
::matrix_type&
stickygpm::stickygpm_regression_data<RealType>
::X() const {
  return _X;
};



template< typename RealType >
const typename stickygpm::stickygpm_regression_data<RealType>
::matrix_type&
stickygpm::stickygpm_regression_data<RealType>
::lsbp_covariates() const {
  return _Z;
};


template< typename RealType >
const typename stickygpm::stickygpm_regression_data<RealType>
::matrix_type&
stickygpm::stickygpm_regression_data<RealType>
::Z() const {
  return _Z;
};




template< typename RealType >
const typename
stickygpm::stickygpm_regression_data<RealType>::matrix_type&
stickygpm::stickygpm_regression_data<RealType>
::mask_locations() const {
  return _image_mask_locations;
};




template< typename RealType >
const typename
stickygpm::stickygpm_regression_data<RealType>::lsbp_index_type&
stickygpm::stickygpm_regression_data<RealType>
::lsbp_random_effects_indices() const {
  return _Z_random_effects_indices;
};



template< typename RealType >
bool stickygpm::stickygpm_regression_data<RealType>
::lsbp_has_global_intercept() const {
  return _Z_has_intercept;
};




template< typename RealType >
void stickygpm::stickygpm_regression_data<RealType>
::lsbp_append_covariates(
  const std::string covariates_file
) {
  matrix_type Ztemp = stickygpm::read_csv<scalar_type>(
    covariates_file
  );
  if ( Ztemp.rows() != n() ) {
    std::ostringstream msg;
    msg << "Improper number of observations ("
	<< Ztemp.rows()
	<< ") from file: "
	<< covariates_file;
    throw std::domain_error( msg.str() );
  }
  if ( _Z.size() == 0 ) {
    _Z = Ztemp;
  }
  else {
    _Z.conservativeResize(
      _Z.rows(),
      _Z.cols() + Ztemp.cols()
    );
    _Z.rightCols( Ztemp.cols() ) = Ztemp;
  }
  std::cout << "Read: " << covariates_file << std::endl;
};



template< typename RealType >
void stickygpm::stickygpm_regression_data<RealType>
::lsbp_append_random_effects(
  const std::string covariates_file
) {
  const int p0 = _Z.cols();
  lsbp_append_covariates( covariates_file );
  _Z_random_effects_indices.push_back(
    Eigen::VectorXi::LinSpaced(_Z.cols() - p0, p0,
			       _Z.cols() - 1)
  );
};





// template< typename RealType >
// void stickygpm::stickygpm_regression_data<RealType>
// ::finalize_lsbp_covariates( const RealType eps ) {
//   _Z = _Zbuff.sparseView( eps, 1 );
//   _Zbuff.setZero(1, 1);
// };



template< typename RealType >
void stickygpm::stickygpm_regression_data<RealType>
::_read_outcome_data(
  const std::vector<std::string>& outcome_files,
  const std::string mask_file
) {
  // std::cout << "File pattern: " << outcome_files_pattern << std::endl;
  // const std::vector<std::string> fnames =
  //   stickygpm::utilities::list_files(outcome_files_pattern);
  // std::cout << "Found " << fnames.size() << " files" << std::endl;
  if ( outcome_files.empty() ) {
    throw std::runtime_error( "No input image files" );
  }
  else if ( (int)outcome_files.size() != n() ) {
    std::ostringstream msg;
    for ( std::string of : outcome_files ) {
      msg << of << "\n";
    }
    msg << "\n";
    msg << outcome_files.size() << " file(s) found "
	<< " (Expected " << n() << ")";
    throw std::runtime_error( msg.str() );
  }
  stickygpm::utilities::progress_bar pb( n() );
  ::nifti_image* mask =
    stickygpm::nifti_image_read(mask_file, 1);
  _image_mask_locations = stickygpm::get_nonzero_xyz( mask )
    .template cast<scalar_type>();
  _Y.conservativeResize( _image_mask_locations.rows(), n() );

  std::cout << "Reading outcome data from " << outcome_files.size()
	    << " files\n";
  for ( int i = 0; i < (int)outcome_files.size(); i++ ) {
    ::nifti_image* data =
      stickygpm::nifti_image_read( outcome_files[i], 1 );
    std::vector<scalar_type> ytemp =
      stickygpm::get_data_from_within_mask<scalar_type>( data, mask );
    _Y.col(i) = Eigen::Map<vector_type>( ytemp.data(), ytemp.size() );
    ::nifti_image_free( data );
    //
    pb++;
    std::cout << pb;
  }
  pb.finish();

  // Print first 10ish file names for sanity check:
  for ( int i = 0; i < std::min((int)10, (int)outcome_files.size());
	i++ ) {
    std::cout << "\t" << outcome_files[i] << std::endl;
  }
  std::cout << "\t..." << std::endl;
  //
  
  ::nifti_image_free( mask );
};


#endif  // _STICKYGPM_REGRESSION_DATA_
