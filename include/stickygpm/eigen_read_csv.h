
#include <Eigen/Core>
#include <string>
#include <vector>

#include "stickygpm/csv_reader.h"


#ifndef _STICKYGPM_EIGEN_READ_CSV_
#define _STICKYGPM_EIGEN_READ_CSV_


namespace stickygpm {


  // This is a somewhat memory inefficient implementation
  //
  template< typename T >
  Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >
  read_csv( const std::string fname ) {
    typedef typename std::vector< std::vector<T> > buffer_type;
    typedef typename Eigen::Matrix<
      T, Eigen::Dynamic, Eigen::Dynamic >
      matrix_type;
    buffer_type b = stickygpm::csv_reader<T>::read_file(fname);
    matrix_type M(b.size(), b[0].size());
    for ( int i = 0; i < (int)b.size(); i++ ) {
      for ( int j = 0; j < (int)b[0].size(); j++ ) {
	M.coeffRef(i, j) = b[i][j];
      }
    }
    return M;
  };
  
  
}




#endif  // _STICKYGPM_EIGEN_READ_CSV_



