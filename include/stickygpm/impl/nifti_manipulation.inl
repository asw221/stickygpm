
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <nifti1_io.h>
#include <string>
#include <stdexcept>
#include <stdio.h>
#include <vector>

#include "stickygpm/defines.h"
#include "stickygpm/utilities.h"







// --- C -------------------------------------------------------------
  

int stickygpm::count_nonzero_voxels(const ::nifti_image* const nii) {
  if (stickygpm::is_float(nii))
    return stickygpm::count_nonzero_voxels_impl<float>(nii);
  else if (stickygpm::is_double(nii))
    return stickygpm::count_nonzero_voxels_impl<double>(nii);
  else
    throw std::logic_error(
      "count_nonzero_voxels: unrecognized image data type");
    
  // Not reached:
  return 0;
};



template< typename ImageType >
int stickygpm::count_nonzero_voxels_impl(const ::nifti_image* const nii) {
  const ImageType* const data_ptr = (ImageType*)nii->data;
  const int nvox = (int)nii->nvox;
  int count = 0;
  ImageType voxel_value;
  for (int i = 0; i < nvox; i++) {
    voxel_value = *(data_ptr + i);
    if (!(isnan(voxel_value) || voxel_value == 0))  count++;
  }
  return count;
};





// --- E -------------------------------------------------------------


template< typename DataType >
void stickygpm::emplace_nonzero_data(
  ::nifti_image* nii,
  const Eigen::Matrix<DataType, Eigen::Dynamic, 1> &nzdat
) {
  if (stickygpm::is_float(nii))
    stickygpm::emplace_nonzero_data_impl<float, DataType>(nii, nzdat);
  else if (stickygpm::is_double(nii))
    stickygpm::emplace_nonzero_data_impl<double, DataType>(nii, nzdat);
  else
    throw std::runtime_error(
      "emplace_nonzero_data: unrecognized image data type");
};



template< typename ImageType, typename DataType >
void stickygpm::emplace_nonzero_data_impl(
  ::nifti_image* nii,
  const Eigen::Matrix<DataType, Eigen::Dynamic, 1> &nzdat
) {
  const int nvox = (int)nii->nvox;
  ImageType* nii_ptr = (ImageType*)nii->data;
  ImageType voxel_value;
  for (int i = 0, j = 0; i < nvox && j < nzdat.size(); i++) {
    voxel_value = (*nii_ptr);
    if (!(isnan(voxel_value) || voxel_value == 0)) {
      (*nii_ptr) = (ImageType)nzdat[j];
      j++;
    }
    ++nii_ptr;
  }
};





// --- G -------------------------------------------------------------


stickygpm::nifti_bounding_box stickygpm::get_bounding_box(
  const ::nifti_image* const nii
) {
  stickygpm::nifti_bounding_box nbb;
  Eigen::MatrixXi ijk = stickygpm::get_nonzero_indices(nii);
  nbb.ijk_min = ijk.colwise().minCoeff();
  nbb.ijk_max = ijk.colwise().maxCoeff();
  nbb.nnz = ijk.rows();
  return nbb;
};





std::vector<int> stickygpm::get_bounding_box_nonzero_flat_index(
  const ::nifti_image* const nii
) {
  if (stickygpm::is_float(nii))
    return stickygpm::get_bounding_box_nonzero_flat_index_impl
      <float>(nii);
  else if (stickygpm::is_double(nii))
    return stickygpm::get_bounding_box_nonzero_flat_index_impl
      <double>(nii);
  else
    throw std::logic_error(
      "get_bounding_box_nonzero_flat_index: unrecognized image data type");
    
  // Not reached:
  return std::vector<int>{};
};


template< typename ImageType >
std::vector<int> stickygpm::get_bounding_box_nonzero_flat_index_impl(
  const ::nifti_image* const nii
) {
  const stickygpm::nifti_bounding_box bb = stickygpm::get_bounding_box(nii);
  const Eigen::Vector3i dims = bb.ijk_max - bb.ijk_min + Eigen::Vector3i::Ones();
  const ImageType* const data_ptr = (ImageType*)nii->data;
  const int nx = nii->nx, ny = nii->ny;  // , nz = nii->nz;
  ImageType voxel_value;
  int nii_index, bounded_index, count = 0;
  std::vector<int> index(dims.prod());
  for (int k = 0; k < dims[2]; k++) {
    for (int j = 0; j < dims[1]; j++) {
      for (int i = 0; i < dims[0]; i++) {
	nii_index = (k + bb.ijk_min[2]) * nx * ny +
	  (j + bb.ijk_min[1]) * nx + (i + bb.ijk_min[0]);
	bounded_index = k * dims[0] * dims[1] + j * dims[0] + i;
	// nii_index = (k + bb.ijk_min[2]) + (j + bb.ijk_min[1]) * nz +
	//   (i + bb.ijk_min[0]) * nz * ny;
	// bounded_index = k + j * dims[2] + i * dims[2] * dims[3];
	voxel_value = *(data_ptr + nii_index);
	if (!(isnan(voxel_value) || voxel_value == 0)) {
	  index[bounded_index] = count;
	  count++;
	}
	else {
	  index[bounded_index] = -1;
	}
      }
    }
  }
  return index;
};






template< typename ResultType >
std::vector<ResultType> stickygpm::get_data_from_within_mask(
  const ::nifti_image* const data,
  const ::nifti_image* const mask
) {
  if ( stickygpm::is_float(data) && stickygpm::is_float(mask) ) {
    return stickygpm::get_data_from_within_mask_impl<
      ResultType, float, float>(data, mask);
  }
  else if ( stickygpm::is_double(data) && stickygpm::is_double(mask) ) {
    return stickygpm::get_data_from_within_mask_impl<
      ResultType, double, double>(data, mask);
  }
  else if ( stickygpm::is_float(data) && stickygpm::is_double(mask) ) {
    return stickygpm::get_data_from_within_mask_impl<
      ResultType, float, double>(data, mask);
  }
  else if ( stickygpm::is_double(data) && stickygpm::is_float(mask) ) {
    return stickygpm::get_data_from_within_mask_impl<
      ResultType, double, float>(data, mask);
  }
  else {
    throw std::domain_error(
      "get_data_from_within_mask: unrecognized image data type");
  }

  // Not reached:
  return std::vector<ResultType>{};
};





template< typename ResultType, typename ImageType, typename MaskType >
std::vector<ResultType> stickygpm::get_data_from_within_mask_impl(
  const ::nifti_image* const nii,
  const ::nifti_image* const mask
) {
  const int nvox_mask = (int)mask->nvox;
  const int nx = nii->nx, ny = nii->ny, nz = nii->nz;
  const stickygpm::qform_type Q =
    stickygpm::qform_matrix(nii).inverse() *
    stickygpm::qform_matrix(mask);
  Eigen::Vector3i ijk = Eigen::Vector3i::Ones();
  Eigen::Vector4f ijk1_mask = Eigen::Vector4f::Ones();
  ImageType* data_ptr = (ImageType*)nii->data;
  MaskType* data_ptr_mask = (MaskType*)mask->data;
  int stride, voxcount = 0;
  std::vector<ResultType> _data;
  // _data.reserve(nvox_mask);
  _data.assign( nvox_mask, 0 );
  for (int k = 0; k < mask->nz; k++) {  // Column-major order
    ijk1_mask.coeffRef(2) = k;
    for (int j = 0; j < mask->ny; j++) {
      ijk1_mask.coeffRef(1) = j;
      for (int i = 0; i < mask->nx; i++) {
	ijk1_mask.coeffRef(0) = i;
	if ( !(isnan(*data_ptr_mask) || (*data_ptr_mask) == 0) ) {
	  ijk = ( Q * ijk1_mask ).head<3>().template cast<int>();
	  if ( (ijk.coeffRef(0) >= 0 && ijk.coeffRef(0) < nx) &&
	       (ijk.coeffRef(1) >= 0 && ijk.coeffRef(1) < ny) &&
	       (ijk.coeffRef(2) >= 0 && ijk.coeffRef(2) < nz)
	       ) {
	    stride = nx * ny * ijk.coeffRef(2) +
	      nx * ijk.coeffRef(1) + ijk.coeffRef(0);
	    // _data.push_back( ResultType(*(data_ptr + stride)) );
	    _data[voxcount] = ResultType(*(data_ptr + stride));
	  }
	  voxcount++;
	}
	++data_ptr_mask;
      }
    }
  }
  // _data.shrink_to_fit();
  return _data;
};
  
  








  

template< typename ResultType >
std::vector<ResultType> stickygpm::get_nonzero_data(
  const ::nifti_image* const nii
) {
  if (stickygpm::is_float(nii))
    return stickygpm::get_nonzero_data_impl<ResultType, float>(nii);
  else if (stickygpm::is_double(nii))
    return stickygpm::get_nonzero_data_impl<ResultType, double>(nii);
  else
    throw std::logic_error("get_nonzero_data: unrecognized image data type");
    
  // Not reached:
  return std::vector<ResultType>{};
};


template< typename ResultType, typename ImageType >
std::vector<ResultType> stickygpm::get_nonzero_data_impl(
  const ::nifti_image* const nii
) {
  const int nvox = (int)nii->nvox;
  std::vector<ResultType> _data;
  _data.reserve(nvox);
  ImageType* data_ptr = (ImageType*)nii->data;
  ResultType voxel_value;
  for (int i = 0; i < nvox; i++) {
    voxel_value = (ResultType)(*(data_ptr + i));
    if (!(isnan(voxel_value) || voxel_value == 0)) {
      _data.push_back(voxel_value);
    }
  }
  _data.shrink_to_fit();
  return _data;
};




Eigen::MatrixXi stickygpm::get_nonzero_indices(
  const ::nifti_image* const nii
) {
  if (stickygpm::is_float(nii))
    return stickygpm::get_nonzero_indices_impl<>(nii);
  else if (stickygpm::is_double(nii))
    return stickygpm::get_nonzero_indices_impl<double>(nii);
  else
    throw std::runtime_error("get_nonzero_indices: unrecognized image data type");
    
  // Not reached:
  return Eigen::MatrixXi::Zero(1, 1);
};


template< typename ImageType >
Eigen::MatrixXi stickygpm::get_nonzero_indices_impl(
  const ::nifti_image* const nii
) {
  if (nii->ndim != 3)
    throw std::logic_error("NIfTI image has dim != 3");
  const int nx = nii->nx, ny = nii->ny, nz = nii->nz, nvox = (int)nii->nvox;
  ImageType* dataPtr = (ImageType*)nii->data;
  std::vector<int> indices;
  for (int i = 0; i < nvox; i++) {
    if (!(isnan(*dataPtr) || *dataPtr == 0)) {
      indices.push_back(i);
    }
    dataPtr++;
  }
  Eigen::MatrixXi ijk(indices.size(), 3);
  for (int i = 0; i < (int)indices.size(); i++) {
    ijk(i, 0) = indices[i] % nx;                // Column-major order
    ijk(i, 1) = (indices[i] / nx) % ny;
    ijk(i, 2) = (indices[i] / (nx * ny)) % nz;
    // ijk(i, 0) = (indices[i] / (nz * ny)) % nx;
    // ijk(i, 1) = (indices[i] / nz) % ny;
    // ijk(i, 2) = indices[i] % nz;
  }
  return ijk;
};



  
Eigen::MatrixXi stickygpm::get_nonzero_indices_bounded(
  const ::nifti_image* const nii
) {
  Eigen::MatrixXi ijk = stickygpm::get_nonzero_indices(nii);
  return (ijk.rowwise() - ijk.colwise().minCoeff());
};








std::vector<int> stickygpm::get_nonzero_flat_index_map(
  const ::nifti_image* const nii
) {
  if (stickygpm::is_float(nii))
    return stickygpm::get_nonzero_flat_index_map_impl<float>(nii);
  else if (stickygpm::is_double(nii))
    return stickygpm::get_nonzero_flat_index_map_impl<double>(nii);
  else
    throw std::runtime_error(
      "get_nonzero_flat_index_map: unrecognized image data type");
    
  // Not reached:
  return std::vector<int>{-1};
};



template< typename ImageType >
std::vector<int> stickygpm::get_nonzero_flat_index_map_impl(
  const ::nifti_image* const nii
) {
  if (nii->ndim < 3)
    throw std::logic_error("NIfTI image has dim < 3");
  const int nvox = (int)nii->nvox;
  ImageType* dataPtr = (ImageType*)nii->data;
  std::vector<int> indices(nvox, -1);
  int j = 0;
  for (int i = 0; i < nvox; ++i, ++dataPtr) {
    if (!(isnan(*dataPtr) || *dataPtr == 0)) {
      indices[i] = j;
      j++;
    }
  }
  return indices;
};








Eigen::MatrixXf stickygpm::get_nonzero_xyz(
  const ::nifti_image* const nii
) {
  if (stickygpm::is_float(nii))
    return stickygpm::get_nonzero_xyz_impl<>(nii);
  else if (stickygpm::is_double(nii))
    return stickygpm::get_nonzero_xyz_impl<double>(nii);
  else
    throw std::runtime_error("get_nonzero_indices: unrecognized image data type");
    
  // Not reached:
  return Eigen::MatrixXf::Zero(1, 1);
};


template< typename ImageType >
Eigen::MatrixXf stickygpm::get_nonzero_xyz_impl(
  const ::nifti_image* const nii
) {
  if (nii->ndim != 3)
    throw std::logic_error("NIfTI image has dim != 3");
  stickygpm::qform_type Q = stickygpm::qform_matrix(nii);
  Eigen::MatrixXf ijk1 =
    stickygpm::get_nonzero_indices_impl<ImageType>(nii)
    .template cast<float>();
  ijk1.conservativeResize(ijk1.rows(), ijk1.cols() + 1);
  ijk1.col(ijk1.cols() - 1) = Eigen::VectorXf::Ones(ijk1.rows());
  Eigen::MatrixXf xyz = ijk1 * Q.transpose();
  xyz.conservativeResize(xyz.rows(), xyz.cols() - 1);
  return xyz;
};




  




// --- I -------------------------------------------------------------



bool stickygpm::is_double(const ::nifti_image* const nii) {
  return stickygpm::nii_data_type(nii) ==
    stickygpm::nifti_data_type::DOUBLE;
};
  


bool stickygpm::is_float(const ::nifti_image* const nii) {
  return stickygpm::nii_data_type(nii) ==
    stickygpm::nifti_data_type::FLOAT;
};


bool stickygpm::is_nifti_file(const std::string &fname) {
  // ::is_nifti_file defined in nifti1_io.h
  const stickygpm::path _initial_path = stickygpm::current_path();
  stickygpm::path fpath(fname);
  bool is_nifti = stickygpm::utilities::file_exists(fname);
  if (is_nifti) {
    stickygpm::current_path(fpath.parent_path());
    is_nifti = (::is_nifti_file(fpath.filename().c_str()) == 1);
    stickygpm::current_path(_initial_path);
  }
  return is_nifti;
};




// --- N -------------------------------------------------------------


std::string stickygpm::nifti_datatype_string(
  const ::nifti_image* const nii
) {
  std::string __dt(::nifti_datatype_string(nii->datatype));
  return __dt;
};



::nifti_image* stickygpm::nifti_image_read(
  const std::string &hname,
  int read_data
) {
  // ::nifti_image_read is defined in nifti1_io.h
  const stickygpm::path _initial_path = stickygpm::current_path();
  stickygpm::path hpath(hname);
  stickygpm::current_path(hpath.parent_path());
  ::nifti_image* _nii = ::nifti_image_read(hpath.filename().c_str(), read_data);
  stickygpm::current_path(_initial_path);
  return _nii;
};


  
void stickygpm::nifti_image_write(
  ::nifti_image* nii,
  std::string new_filename
) {
  // ::nifti_image_write is defined in nifti1_io.h
  const stickygpm::path _initial_path = stickygpm::current_path();
  if (new_filename.empty()) {
    new_filename = std::string(nii->fname);
  }
  stickygpm::path hpath(new_filename);
  stickygpm::current_path(hpath.parent_path());
  remove(hpath.filename().c_str());
  ::nifti_set_filenames(nii, hpath.filename().c_str(), 1, 1);
  ::nifti_image_write(nii);
  stickygpm::current_path(_initial_path);
};

  

stickygpm::nifti_data_type stickygpm::nii_data_type(
  const ::nifti_image* const nii
) {
  stickygpm::nifti_data_type __dt = stickygpm::nifti_data_type::OTHER;
  try {
    __dt = static_cast<stickygpm::nifti_data_type>(nii->datatype);
  }
  catch (...) { ; }
  return __dt;
};





// --- Q -------------------------------------------------------------
  

stickygpm::qform_type stickygpm::qform_matrix(
  const ::nifti_image* const img
) {
  std::vector<float> m(16);
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      m[i * 4 + j] = img->qto_xyz.m[i][j];
  return Eigen::Map<stickygpm::qform_type>(m.data());
};





// --- S -------------------------------------------------------------


bool stickygpm::same_data_types(
  const ::nifti_image* const first_img,
  const ::nifti_image* const second_img
) {
  return (static_cast<stickygpm::nifti_data_type>
	  (first_img->datatype) ==
	  static_cast<stickygpm::nifti_data_type>
	  (second_img->datatype));
};




stickygpm::qform_type stickygpm::sform_matrix(
  const ::nifti_image* const img
) {
  std::vector<float> m(16);
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      m[i * 4 + j] = img->sto_xyz.m[i][j];
  return Eigen::Map<stickygpm::qform_type>(m.data());
};





// --- T -------------------------------------------------------------


template<typename RealType>
void stickygpm::threshold( ::nifti_image * const nii, const RealType cut ) {
  if (stickygpm::is_float(nii))
    stickygpm::threshold_impl<float>(nii, (float)cut);
  else if (stickygpm::is_double(nii))
    stickygpm::threshold_impl<double>(nii, (double)cut);
  else
    throw std::logic_error(
      "count_nonzero_voxels: unrecognized image data type");
};


template<typename DataType>
void stickygpm::threshold_impl( ::nifti_image * const nii, const DataType cut ) {
  const int nvox = (int)nii->nvox;
  DataType* data_ptr = (DataType*)nii->data;
  for (int i = 0; i < nvox; ++i, ++data_ptr) {
    if (isnan(*data_ptr)) {
      *data_ptr = (DataType)0;
    }
    else if (std::abs(*data_ptr) < cut) {
      *data_ptr = (DataType)0;
    }
  }
};



// --- V -------------------------------------------------------------


Eigen::Vector3f stickygpm::voxel_dimensions(
  const ::nifti_image* const nii
) {
  Eigen::Vector3f dims_;
  dims_ << nii->dx, nii->dy, nii->dz;
  return dims_;
};



Eigen::Vector3f stickygpm::voxel_dimensions(
  const stickygpm::qform_type &Q
) {
  const Eigen::Matrix<float, 4, 3> I = Eigen::Matrix<float, 4, 3>::Identity();
  return (Q * I).colwise().norm();
};





  
