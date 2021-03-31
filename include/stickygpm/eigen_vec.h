
#include <Eigen/Core>


#ifndef _STICKYGPM_EIGEN_VEC_
#define _STICKYGPM_EIGEN_VEC_


namespace stickygpm {

  template< typename T, int _Rows, int _Cols,
	    int _Options, int _MaxCompTimeRows, int _MaxCompTimeCols >
  Eigen::Matrix<T, Eigen::Dynamic, 1> vec(
    const Eigen::Matrix<T, _Rows, _Cols, _Options,
    _MaxCompTimeRows, _MaxCompTimeCols>& M
  ) {
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, 1>
      vector_type;
    const int n = M.rows(), p = M.cols();
    vector_type v(n * p);
    int vi = 0;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < p; j++) {
	v[vi] = M.coeffRef(i, j);
	vi++;
      }
    }
    return v;
  };
  
}


#endif  // _STICKYGPM_EIGEN_VEC_
