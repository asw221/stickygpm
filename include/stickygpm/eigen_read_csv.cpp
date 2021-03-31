
#include <Eigen/Core>
#include <string>
#include <vector>

#include "stickygpm/csv_reader.h"



namespace stickygpm {


  // This is a fairly memory inefficient implementation
  //
  template< typename T, int _Rows, int _Cols, int _Options,
	    int _MaxCompRows, int _MaxCompCols>
  Eigen::Matrix<T, _Rows, _Cols, _Options, _MaxCompRows, _MaxCompCols>
  read_csv(const std::string fname) {
    typedef typename std::vector< std::vector<T> > buffer_type;
    typedef typename Eigen::Matrix<
      T, _Rows, _Cols, _Options, _MaxCompRows, _MaxCompCols >
      matrix_type;
    typedef typename Eigen::Map<
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
		    Eigen::RowMajor> >
      mapped_buffer_type;
    buffer_type b = stickygpm::csv_reader<T>::read_file(fname);
    matrix_type M =
      mapped_buffer_type(b.data(), b.size(), b[0].size());
    return M;
  };
  
  
}








